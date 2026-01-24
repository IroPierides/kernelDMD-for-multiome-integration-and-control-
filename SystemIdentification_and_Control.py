# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from matplotlib.patches import Rectangle, Patch
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm
from scipy.linalg import solve_continuous_lyapunov as lyap
import cvxpy as cp
from scipy.linalg import solve_discrete_are, sqrtm
import os
from sklearn.metrics import r2_score
from reconstructions import *
from matplotlib import rcParams
from matplotlib import font_manager as fm
times_font = fm.FontProperties(fname="times.ttf")
import matplotlib 
import numpy as np
from scipy.linalg import solve_continuous_lyapunov as solve_lyap
from scipy.linalg import svd, eigh, sqrtm

#np.random.seed(8)  
#os.environ["OMP_NUM_THREADS"] = "1" 

feature_labels = pd.read_csv("feature_table.csv")
feature_labels = feature_labels[~feature_labels["Name"].isna()].reset_index(drop=True)

def kernel_gradient(kernel, inputs, gamma, coef0, degree, gamma1, delta1, coef2, coef3, adj_matrix):
    """
    kernel calculation and their respective gradient

    Parameters:
    - kernel: specify linear, rbf, polynomial or sigmoid kernel
    - inputs (numpy array): concatenated inputs
    - gamma: coefficient of the vector inner product for rbf and polynomial kernels 
    - coef0: constant offset added to scaled inner product for polynomial kernel
    - degree: kernel degree for polynomial kernel 
    - gamma1: coefficient of the vector inner product for sigmoidal kernels 
    - delta1: constant offset subtracted to scaled inner product for sigmoidal kernel
    - coef2: weight term for linear part of sigmoidal kernel 
    - coef3: weight term for sigmoid part of sigmoidal kernel 
    - adj_matrix: adjacency matrix for concatenated input data (apriori pathway information)
        
    Returns:
    - grad: gradient of kernel 
    - kernOut: the kernel
    
    """
    fixed_point = np.mean(inputs, axis=0)
    
    if kernel == "linear":
        grad = inputs 
        kernOut = pairwise_kernels(inputs, metric=kernel)

    elif kernel == "rbf":
        centered = inputs - fixed_point  
        sq_norm = np.linalg.norm(centered, axis=1) ** 2
        exp_term = np.exp(-gamma * sq_norm)
        grad = -2 * gamma * (exp_term[:, None] * centered)  
        kernOut = pairwise_kernels(inputs, metric=kernel, gamma=gamma)

    elif kernel == "poly":
        dot_products = inputs @ fixed_point
        grad = gamma * degree * (coef0 + gamma * dot_products)[:, None] ** (degree - 1) * fixed_point
        kernOut = pairwise_kernels(inputs, metric=kernel, gamma=gamma, coef0=coef0, degree=degree)

    elif kernel == "sigmoid":
        np.fill_diagonal(adj_matrix, 0)
        D = np.diag(np.sum(adj_matrix, axis=1)) # degree matrix
        L = D - (adj_matrix) # Laplacian matrix 
        L += 1e-5 * np.eye(L.shape[0]) 
        X_dot = inputs @ fixed_point - delta1  
        if inputs.shape[1] == 1: 
            X_dot_reg = X_dot
            sigmoid = 1 / (1 + np.exp(-gamma1 * X_dot_reg))  
            grad = coef3 * gamma1 * (sigmoid * (1 - sigmoid))[:, None] * fixed_point[None, :] - coef2 * inputs 
            kernOut = coef3 * sigmoid[:, None] - coef2 * (inputs @ inputs.T) 
        else:
            z = gamma1 * inputs @ (np.eye(L.shape[0]) - L) @ fixed_point - delta1 
            sigmoid = 1 / (1 + np.exp(-z))  
            grad = coef3 * gamma1 * sigmoid * (1 - sigmoid)[:, None] @ ((np.eye(L.shape[0]) - L).T @ inputs.T).T - coef2 * inputs 
            kernOut = coef3 * sigmoid[:, None] - coef2 * (inputs @ inputs.T) 
        
    return grad, kernOut

def cluster_nodes_based_on_edges(graph):
    """
    Cluster nodes based on edge sign and weight similarity.
    
    Args:
    - graph (networkx.Graph): The graph containing nodes and weighted edges.
    
    Returns:
    - labels (list): List of cluster labels for each node.
    - clustering: The clustering model used (AgglomerativeClustering or other).
    """
    
    node_features = {}
    for node in graph.nodes:
        node_features[node] = []
        for neighbor in graph.neighbors(node):
            weight = graph[node][neighbor]['weight']
            sign = np.sign(weight)  
            abs_weight = np.abs(weight)  
            node_features[node].append([sign, abs_weight])  
            
    node_features_matrix = []
    for node in graph.nodes:
        features = np.array(node_features[node]).flatten()
        node_features_matrix.append(features)
    
    node_features_matrix = np.array(node_features_matrix)
    
    scaler = StandardScaler()
    node_features_matrix_scaled = scaler.fit_transform(node_features_matrix)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(node_features_matrix_scaled)
    
    return labels, clustering


def modes_heatmap(Phi1, mode_pairs1, Phi4, mode_pairs4, evals1, evals4, names, adj_matrix, products, col_ind1, col_ind2, CO2_ind1, CO2_ind2):
    """
    Creates eigenmode graphs based on inner eigenmode products after applying a threshold

    Parameters:
    - Phi1, Phi4: eigenmodes of each species
    - mode_pairs1, mode_pairs4: mode pairs of each species
    - evals1, evals4: eigenvalues
    - names: feature names   
    
    """
    mode_index1 = [mode_pairs1[k][0] for k in range(len(mode_pairs1))]
    mode_index4 = [mode_pairs4[k][0] for k in range(len(mode_pairs4))]

    evals1 = evals1[mode_index1]
    evals4 = evals4[mode_index4]

    Phi1 = Phi1[:, mode_index1]
    Phi4 = Phi4[:, mode_index4]
    
    Phi1 = np.abs(Phi1)
    Phi4 = np.abs(Phi4)
    
    Phi1 = Phi1[:, col_ind1]
    Phi4 = Phi4[:, col_ind2]
    
    kmax = 60
    kmax_inds1 = []
    for ii in range(Phi1.shape[1]):
        kmax_inds1.append(Phi1[:, ii].argsort()[-kmax:])

    kmax = 60
    kmax_inds4 = []
    for ii in range(Phi4.shape[1]):
        kmax_inds4.append(Phi4[:, ii].argsort()[-kmax:])
    
    G1_list= []
    G2_list = []
    
    for i in range(Phi1.shape[1]):
        feature_index = kmax_inds1[i]  # List of top-k indices for mode i      
        feature_names = np.asarray(names) #[feature_index]  # Subset of feature names
        exclude_ind = [i for i in range(Phi1.shape[0]) if i not in feature_index]
        mode1 = Phi1[:, i]
        mode1[exclude_ind, ] = 0
        sorted_mode_copy = mode1  # Only top-k values
        adj1 = adj_matrix[:, :]
        adj1 = adj1[:, :]
        Phi_G1 = np.outer(sorted_mode_copy, sorted_mode_copy) 
        df = pd.DataFrame(Phi_G1, index=feature_names, columns=feature_names)
        df.index.name = 'Target'

        if i in CO2_ind1:
            G1_list.append(df)

    for i in range(Phi4.shape[1]):
        feature_index = kmax_inds4[i]   
        feature_names = np.asarray(names) #[feature_index]  # Subset of feature names
        exclude_ind = [i for i in range(Phi4.shape[0]) if i not in feature_index]
        mode2 = Phi4[:, i]
        mode2[exclude_ind, ] = 0
        sorted_mode_copy2 = mode2
        adj2 = adj_matrix[:, :]
        adj2 = adj2[:, :]
        Phi_G2 =  np.outer(sorted_mode_copy2, sorted_mode_copy2) 
        df = pd.DataFrame(Phi_G2, index=feature_names, columns=feature_names)
        df.index.name = 'Target'
        
        if i in CO2_ind2:
            G2_list.append(df)
            
    return G1_list, G2_list

def network_output_files(label, C, K_full, nT, inputs, modes, eVals, mode_pairs, features, feat_type, names, products, col_ind1):
    """
    Feature information: ranking on each eigenmode, spectral eigenmode clustering, pathway, observability rank
    Creates spectral cluster heatmap with feature timeseries concentration levels
    eigenvalue classification

    Parameters:
    - label: species label
    - C (numpy array): output matrix
    - K_full (numpy array): Koopman matrix 
    - nT: time horizon 
    - inputs: input data
    - modes: eigenmodes 
    - eVals: eigenvalues
    - mode_pairs: conjugate eigenmode pairs
    - features: orthogroup names 
    - feat_type: type of feature (transcript, protein, metabolite)
    - names: feature names   
    - products: feature full names   
    
    Returns:
    - df_up_new: dataframe with feature information and cluster membership
    
    """
    clean_filename = re.sub(r'[\$\{\}\\]', '', label)
    clean_filename = re.sub(r'\s+', '_', clean_filename)
    clean_filename = re.sub(r'[^a-zA-Z0-9_.]', '', clean_filename)
    
    eig, eigen_vec = np.linalg.eig(K_full)
    sorted_inds = np.argsort(-np.abs(eig))
    eig = eig[sorted_inds]
    eigen_vec = eigen_vec[:, sorted_inds]
    for i, val in enumerate(eig):
        if np.abs(np.real(val)) - 1 < 0.001: 
            eigenvector=eigen_vec[:, i]
            break
    
    # normalize eigenvector so that its values sum up to 1
    eigenvector = np.abs(eigenvector)
    eigenvector = eigenvector / np.sum(eigenvector)
    
    modes_index = [mode_pairs[k][0] for k in range(len(mode_pairs))]
    Phi = modes[:, modes_index]
    Phi = Phi[:, col_ind1]
    real_Phi = np.real(Phi)
    imag_Phi = np.imag(Phi)
    Phi_concat = np.concatenate((real_Phi, imag_Phi), axis=1)
    best_score = -1
    best_n_clusters = 5 
    
    for n_clusters in range(2, min(11, len(Phi_concat))):  # test 2 to 10 clusters
        spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',  # or 'rbf', or precomputed
        n_neighbors=20,
        assign_labels='cluster_qr',
        random_state=42
    )
        labels = spectral.fit_predict(Phi_concat)
        
        score = silhouette_score(Phi_concat, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
        
    spectral = SpectralClustering(
    n_clusters=best_n_clusters,
    affinity='nearest_neighbors',  
    n_neighbors=20,
    assign_labels='cluster_qr',
    random_state=42
    )
    clusters = spectral.fit_predict(Phi_concat)
    
    # Observability Gramian
    CtC = np.matmul(C.T, C)
    Xo = np.zeros_like(K_full)
    for ii in range(nT):
        A_pow = np.linalg.matrix_power(K_full, ii)
        Xo += np.matmul(np.matmul(A_pow, CtC), A_pow.T)

    D, V = np.linalg.eig(Xo)
    W = V[:, 0:1]
    W = W.T
    sorted_inds = list(np.argsort(np.abs(W[0, :])))
    Wsorted = W[:, sorted_inds]

    # Feature ranks by observability
    df_up_new = pd.DataFrame({
        'feature': features,
        'feature name': names,
        'feature_prod': products,
        'feat_type': feat_type
    })
    
    pathway_dict = {val: feature_labels['Pathway'].iloc[idx] for idx, val in enumerate(feature_labels['Name'])}
    
    met_names = pd.read_csv('met_names.csv').iloc[:, 0].values

    for key in met_names:
        pathway_dict[key] = "Metabolome" 
    
    df_up_new['pathway'] = df_up_new['feature name'].str.replace(r"\(.*\)", "", regex=True).str.strip().map(pathway_dict)
    df_up_new['pathway'] = df_up_new['pathway'].fillna('Unknown')
    rank_tag_inds = [len(W[0]) - list(Wsorted[0]).index(W[0][ii]) for ii in df_up_new.index]
    rank_tag_inds_per = [100 - np.round((rank - 1) / len(W[0]) * 100, 2) for rank in rank_tag_inds]
    df_up_new['obs_rank'] = np.round(rank_tag_inds_per, 0)
    
    modes = modes[:, modes_index]
    modes = modes[:, col_ind1]
    
    with open('modes.txt', 'w') as f:
        f.write(f'modes shape: {modes.shape}')
                
    df_up_new['mode_cluster'] = 'na'
    # df_up_new['eigenvalue'] = 'na'
    kmax = 90
    kmax_inds_list = []
    for i in range(modes.shape[1]):
        kmax_inds = np.abs(modes[:, i]).argsort()[-kmax:]
        kmax_inds_list.append(kmax_inds)

    df_up_new['mode_cluster'] = [[] for _ in range(len(df_up_new))]
   # df_up_new['eigenvalue'] = [[] for _ in range(len(df_up_new))]
    df_up_new['eigenmode_rank'] = [[] for _ in range(len(df_up_new))]
    
    for idx in range(modes.shape[1]):
        for rank, feature_idx in enumerate(kmax_inds_list[idx]):
            df_up_new.at[feature_idx, 'mode_cluster'].append(idx + 1)
            # df_up_new.at[feature_idx, 'eigenvalue'].append(eVals[idx])
            df_up_new.at[feature_idx, 'eigenmode_rank'].append(rank)

    df_up_new = df_up_new.reset_index(drop=True)
    
    df_up_new['spectral_cluster'] = clusters + 1
    
    scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
    inputs_df = pd.DataFrame(scaler.fit_transform(inputs.T[:, :12]))
    inputs_df.columns = np.repeat(['4:00', '8:00', '13:00', '19:00'], 3)
    inputs_df = inputs_df.groupby(inputs_df.columns, axis=1).mean()
    desired_order = ['4:00', '8:00', '13:00', '19:00']
    inputs_df = inputs_df[sorted(inputs_df.columns, key=lambda x: desired_order.index(x))]

    df_up_new = df_up_new.reset_index(drop=True)
    inputs_df = inputs_df.reset_index(drop=True)
    df_up_new = pd.concat((df_up_new, inputs_df), axis=1)
    df = df_up_new.set_index('feature name')

    df['mode_cluster'] = df['mode_cluster'].apply(lambda x: tuple(x) if isinstance(x, list) and x else ('unknown',))
    df_sorted = df.sort_values(by=['spectral_cluster', 'pathway'])
    
    unique_modes = pd.Series(df_sorted['mode_cluster']).drop_duplicates().tolist()
    mode_palette = sns.color_palette("tab10", len(unique_modes))
    mode_color_map = dict(zip(unique_modes, mode_palette))

    unique_pathways = df_sorted['pathway'].unique()
    pathway_palette = sns.color_palette("tab20", len(unique_pathways))
    pathway_color_map = dict(zip(unique_pathways, pathway_palette))

    timepoint_cols = ['4:00', '8:00', '13:00', '19:00']
    spectral_clusters = df_sorted['spectral_cluster'].unique()
    fig = plt.figure(figsize=(5, 15))
    gs = gridspec.GridSpec(nrows=len(spectral_clusters), ncols=1,
        height_ratios=[len(df_sorted[df_sorted['spectral_cluster'] == c]) * 10 for c in spectral_clusters],
        left=0.4, right=0.6, hspace=0.5)

    axes = [fig.add_subplot(gs[i]) for i in range(len(spectral_clusters))]

    for ax, cluster in zip(axes, spectral_clusters):
        cluster_df = df_sorted[df_sorted['spectral_cluster'] == cluster]
        heatmap_data = cluster_df[timepoint_cols]

        sns.heatmap(
            heatmap_data, norm=PowerNorm(gamma=0.3), ax=ax, cmap="viridis", linewidths=0.2,
            linecolor='gray', cbar=False, yticklabels=True, xticklabels=True
        )

        ax.tick_params(axis='y', labelsize=2, labelright=True, labelleft=False)
        ax.tick_params(axis='x', labelsize=2, labelright=False, labelleft=False, rotation=0)

        for label in ax.get_yticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment('left')
        ax.set_ylabel(None)
        ax.set_title(f"Spectral Cluster {cluster}", fontsize=3, loc='left')

        for i, (_, row) in enumerate(cluster_df.iterrows()):
            ax.add_patch(Rectangle((-0.65, i), 0.2, 1, color=pathway_color_map[row['pathway']], transform=ax.transData, clip_on=False))
            ax.add_patch(Rectangle((-0.85, i), 0.2, 1, color=mode_color_map[row['mode_cluster']], transform=ax.transData, clip_on=False))

    visible_modes = df_sorted['mode_cluster'].unique()
    mode_legend = [Patch(facecolor=mode_color_map[m], label=str(m)) for m in visible_modes]
    pathway_legend = [Patch(facecolor=color, label=label) for label, color in pathway_color_map.items()]
    fig.legend(
        handles=pathway_legend,
        loc='lower right',
        bbox_to_anchor=(1.0, 0.01),
        title='Legend',
        fontsize=3,
        title_fontsize=5,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )


    cbar_ax = fig.add_axes([0.82, 0.91, 0.15, 0.012])  

    norm = plt.Normalize(vmin=df_sorted[timepoint_cols].min().min(),
                        vmax=df_sorted[timepoint_cols].max().max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Expression level', fontsize=5)  
    cbar.ax.tick_params(labelsize=4)  
    fig.suptitle(
        f"{clean_filename}",
        fontsize=12,
        y=0.98  
    )
    plt.tight_layout(rect=[0.3, 0.05, 0.75, 0.9])
    plt.show()
    plt.savefig(f'spectral_cluster_distributions_{clean_filename}.pdf', bbox_inches='tight')
    
    attractor_info = []
    type = None
    for i, eigval in enumerate(eVals):
        eig_abs = np.round(np.abs(eigval), 2)
        real_part = np.round(np.real(eigval), 2)
        imag_part = np.round(np.imag(eigval), 2)

        if real_part <= 1 and real_part > 0.9 and np.abs(imag_part) < 0.1:
            type = 'steady state'
        
        if real_part < -0.9 and imag_part == 0:
            type = 'Period-2 oscillation'
        
        if np.abs(real_part) < 1 and eig_abs > 0.85:
            type = 'chaotic/cyclical oscillations'
        
        # Stable: Eigenvalue inside or on the unit circle
        if real_part <= 1 and eig_abs <= 1:
            type = 'stable'
        
        # Unstable: Eigenvalue outside the unit circle (modulus > 1)
        if eig_abs > 1:
            type = 'unstable'
        
        # Decaying: Eigenvalue inside the unit circle and real part < 0
        if np.abs(real_part) < 0.8 and eig_abs < 0.85:
            type = 'decaying'

        top_features = np.zeros(len(features), dtype=int)
     #   top_features[kmax_inds_list[i]] = 1
    
    df_up_new.to_csv(f'{clean_filename}_mode_clusters.csv')
    
    return df_up_new


def gramian_hsv_ranking(A, B, C, k, tol=1e-6):
    """
    ranking of features for data reduction via Hankel Singular Values 

    Parameters:
    - A (numpy array): linear (Koopman) operator
    - B (numpy array): control input matrix 
    - C (numpy array): output matrix 
    - k: number of features to retain, also 
    - tol: tolerance level to threshold HSV values

    Returns:
    - rankings: HSV feature ranks 
    - eigs: HSVs
    
    """
    
    Wc = lyap(A, -B @ B.T)
    rankings = []
    eigs = []

    
    Wo = lyap(A.T, -C.T @ C)
    hsv = np.sqrt(np.abs(np.linalg.eigvals(Wc @ Wo)))

    sort_indices = np.argsort(-hsv) 
    sorted_hsv = hsv[sort_indices]

    stable_rank = []
    retained_hsvs= []
    used = set()
    for idx in sort_indices:
        if len(stable_rank) >= k:
            break
        if all(np.abs(hsv[idx] - hsv[j]) > tol for j in stable_rank):
                stable_rank.append(idx)
                retained_hsvs.append(hsv[idx])
        
    rankings = np.array(stable_rank)
    eigs = hsv
        
    return rankings, np.asarray(retained_hsvs)

def linearMPC(B, C1, K1, inputs1, inputs2, Pheno1, Pheno2, n_phenos, n_controls, features_kept, with_fig=True):
    """
    linear Model Predictive Control (MPC) for phenocopying outputs 

    Parameters:
    - B (numpy array): control input matrix 
    - C1 (numpy array): output matrix 
    - K1 (numpy array): Koopman matrix
    - inputs1, inputs2: inputs of species 1 and 2
    - Pheno1, Pheno2: outputs of species 1 and 2
    - n_phenos: number of phenotypes
    - n_controls: number of input controls
    - with_fig = whether to produce a figure of the reconstructed outputs
    
    Returns:
    - x_input_array: optimized state input array for species 1
    - u_input_array: optimized control input array for species 1
    - sensitivity_matrix: sensitivity values for each feature
    - reconstruction accuracies of the C. rosea outputs
    
    """
    N = 12  # prediction horizon
    x_0 = inputs1[0, :].T
    y_0 = Pheno1.T[:, 0]
    n_states = inputs1.shape[1]
    x_inputs = cp.Variable((n_states, N + 1))
    zref = Pheno2.T 

    x_input_array = np.zeros((n_states, N + 1))
    z_output_array = np.zeros((zref.shape[0], zref.shape[1]))
    u_inputs = cp.Variable((n_controls, N))  # Control inputs
    z = cp.Variable((n_phenos, N + 1,))  # Output trajectory

    Q = np.eye(n_phenos) * 50 
    R_x = np.eye(n_states) 
    R_c = np.eye(n_controls)

    # Cost function setup
    costlist = 0.0
    constrlist = [x_inputs[:, 0] == x_0, z[:, 0] == y_0]  # Initial condition constraints
    
    P = solve_discrete_are(K1, B, R_x, R_c)
    K = np.linalg.inv(R_c + B.T @ P @ B) @ B.T @ P @ K1                  

    for i in range(N):
        u_diff = -K @ (x_inputs[:, i] - inputs2.T[:, i])
        constrlist += [u_inputs[:, i] == u_diff]
            
        if i > 0:
            costlist += 0.5 * cp.quad_form(z[:, i] - zref[:, i] , Q)  # Output cost with weight
        costlist += 0.5 * cp.quad_form(u_inputs[:, i], R_c)  # Control cost with weight
        costlist += 0.5 * cp.quad_form(x_inputs[:, i], R_x)  # State cost with weight
        if i + 1 < N:
            constrlist += [x_inputs[:, i + 1] == K1 @ x_inputs[:, i] + B @ (u_inputs[:, i])]
        if i < N - 1:
            constrlist += [z[:, i + 1] == C1 @ (K1 @ x_inputs[:, i])]

    costlist += 0.5 * cp.quad_form(z[:, N - 1] - zref[:, N], np.eye(n_phenos) ) + \
                    0.5 * cp.quad_form(x_inputs[:, N], np.eye(n_states) ) + \
                    0.5 * cp.quad_form(u_inputs[:, N - 1], np.eye(n_controls))  # Last step cost

    prob = cp.Problem(cp.Minimize(costlist), constrlist)
    prob.solve(solver='CLARABEL', verbose=False)

    nominal_cost = prob.value 
    
    if x_inputs.value is not None:
        x_input_array = x_inputs.value
    if u_inputs.value is not None:
        u_input_array = u_inputs.value
    if z.value is not None:
        z_output_array = z.value

    # Initialize sensitivity matrix
    sensitivity_matrix = np.zeros((n_states, n_controls, N))

    for i in range(N):
        # Sensitivity of the state trajectory with respect to control input u_i
        state_sensitivity = np.linalg.pinv(K1 - B @ K) @ B
        sensitivity_matrix[:, :, i] = state_sensitivity

    sensitivity_matrix = np.mean(sensitivity_matrix, axis=1)
    
    control_sensitivity = pd.DataFrame(sensitivity_matrix, index=features_kept)
    
    outputs1, cd, _, r2 = reconstructed_outputs_MPC(K=K1, C=C1, outs=Pheno2.T, inputs=x_input_array, B=B, u=u_input_array, ntimepts=12, n_phenos=n_phenos)
        
    if with_fig:
        
        labels = ['Malic acid\n (z-score)', u'H\u2082O\n (z-score)', u'CO\u2082\n (z-score)', 'PCK1\n (z-score)', 'PPC1\n (z-score)', 'PHO1\n (z-score)', 'PPD\n (z-score)']

        # plot control reconstruction figures
        rcParams['font.family'] = 'Arial'
        rcParams['font.size'] = 12
        matplotlib.rcParams['axes.grid'] = False

        my_figsize = (12, 5) 
        corr_tspan = range(0, 12, 1)

        fig1, axs1 = plt.subplots(2, 4, figsize=my_figsize)
        fig1.patch.set_facecolor('white')
        fig1.tight_layout(pad=2.0)  
        
        axs1 = axs1.flatten()
        
        
        for i in range(len(labels)):
            axs1[i].plot(corr_tspan, Pheno2.T[i, :12], label=r"$\it{C.\ rosea}$")
            axs1[i].plot(corr_tspan, outputs1[i, :12], color="teal",
                            label=r"$\it{C.\ major}$ OC:" + str(round(cd[i], 2)) + " R\u00b2")
                
            axs1[i].set_xticks(range(0, 12, 1))
            axs1[i].set_xticklabels(['4:00', '8:00', '13:00', '19:00'] * 3, fontproperties=times_font, rotation=45)
            axs1[i].tick_params(axis='both', labelsize=8)
            axs1[i].set_ylabel(labels[i], fontsize=8, fontproperties=times_font)
            axs1[i].legend(loc='best', prop={'size': 8})

        fig1.delaxes(axs1[7]) 
        plt.tight_layout(pad=3)
            
        plt.savefig(f"Figure 6.eps", format='eps', dpi=600)
        plt.close(fig1)

    return x_input_array, u_input_array, control_sensitivity, cd

def cross_validation(inputs, inputs_t, n_phenos, rank, Pheno, param0, param1, param2, param3, param4, param7):
    
    """
    leave-3-out cross validation of the sigmoid kernel to check for overfitting

    Parameters:
    - inputs: inputs 
    - inputs_t: inputs shifted one time point ahead
    - Pheno: outputs 
    - n_phenos: number of phenotypes
    - rank: number of modes to keep
    - param0-7: relevant parameters for kernels 
    
    Returns:
    - outputs a text file with mean accuracies for input and output distributions
    
    """
    
    n_repeats = 10

    accuracies = []; accuracies_outs = []

    for _ in range(n_repeats):
    
        train_idx = np.arange(0, 20)
        test_idx = np.arange(20, 23)

        inputs_train = inputs[train_idx]
        inputs_train_t = inputs_t[train_idx]

        inputs_test = inputs[test_idx]
        inputs_test_t = inputs_t[test_idx]

            # Kernel training
        beta = - param4 * np.mean(np.dot(inputs_train, inputs_train.T))
        coef2 = 0.1 * np.var(inputs_train)
                
        grad_train, kern_train = kernel_gradient(
            kernel=param0, inputs=inputs_train,
            gamma=param1, coef0=param2,
            degree=param3, gamma1=param4,
            delta1=beta, coef2=coef2,
                coef3=param7
            )
            
        kern_train += 1e-3 * np.eye(kern_train.shape[0])
            
        U, _, _ = np.linalg.svd(grad_train.T)
        U = U[:, :rank]

        W1 = cp.Variable((inputs_train.shape[1], len(train_idx)))

        C = cp.Variable((n_phenos, len(train_idx)))

        K_tilde = U.T @ W1 @ grad_train @ U
        constraints = [cp.norm(K_tilde, 2) - 1 <= 0.001]

        outs = Pheno.T[:, train_idx]
        outs_test = Pheno.T[:, test_idx]
        obj = cp.Minimize(
                cp.square(cp.norm(inputs_train_t.T - W1 @ kern_train, "fro")) +
                3 * (1 - 0.99 ) * cp.norm(W1, "fro") + 3 * 0.99 * cp.mixed_norm(W1, 2, 1) +
                cp.square(cp.norm(outs - C @ kern_train, "fro"))
            )

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver='CLARABEL')
            W1 = W1.value
            C = C.value
        except cp.error.SolverError:
            continue

        coef2 = 0.1 * np.var(inputs_test)

        bias_test = np.mean(inputs_test, axis=0)
        
        X_dot = inputs_test @ inputs_train.T - param4 * np.mean(np.dot(inputs_test, inputs_test.T))  
        sigmoid = 1 / (1 + np.exp(-param4 * X_dot))
        sigmoid_kernel = param7 * sigmoid - coef2 * (inputs_test @ inputs_train.T)
        sigmoid_kernel += 1e-3 * np.eye(sigmoid_kernel.shape[0], sigmoid_kernel.shape[1])

        bias_test = bias_test[None, :]
        snapshots = W1 @ sigmoid_kernel.T
            
        snapshots_train = W1 @ kern_train
        outputs = C @ sigmoid_kernel.T
            
        error = np.linalg.norm(inputs_test_t.T - snapshots)
        accuracy = 1 - error / np.linalg.norm(inputs_test_t)
        accuracy =  r2_score(inputs_test_t.T, snapshots)

        error = np.linalg.norm(inputs_train_t.T - snapshots_train) 
        accuracy_outs = r2_score(outs_test, outputs)
        accuracies.append(accuracy)
        accuracies_outs.append(accuracy_outs)
            
    mean_accuracy = np.mean(accuracies)
    mean_accuracy_outs = np.mean(accuracies_outs)

    with open('accuracy2.txt', 'w') as f:
        f.write(f'accuracy: {mean_accuracy}')
        f.write(f'accuracy outs: {mean_accuracy_outs}')
        
def controllability_gramian(A, B):
    return solve_lyap(A, -B @ B.T)

def observability_gramian(A, C):
    return solve_lyap(A.T, -C.T @ C)

def hsv_basis(A, B, C, tol=1e-6):
    """
    Compute balanced basis using Hankel singular values (HSV).
    """
    Wc = controllability_gramian(A, B)
    Wo = observability_gramian(A, C)

    # Force symmetry
    Wc = 0.5 * (Wc + Wc.T)
    Wo = 0.5 * (Wo + Wo.T)

    sqrtWo = sqrtm(Wo)
    M = sqrtWo @ Wc @ sqrtWo
    svals, Phi = eigh(M)
    svals = np.real(svals)
    idx = np.argsort(svals)[::-1]
    svals = svals[idx]
    Phi = Phi[:, idx]
    hsv = np.sqrt(np.abs(svals))
    
    hsv = hsv[:20]
    Phi = Phi[:, :20]

    T = Wc @ sqrtWo @ Phi @ np.diag(1.0 / np.sqrt(hsv))
    return T, hsv

def intersection_basis(U, V, sv_threshold=1-1e-6):
    M = U.T @ V
    P, S, Qh = svd(M, full_matrices=False)
    shared_idx = np.where(S > sv_threshold)[0]
    if len(shared_idx) == 0:
        return np.zeros((U.shape[0], 0))
    return U @ P[:, shared_idx]

    
def system_reIdendification(inputs, inputs_t, coef, adj_matrix_kept, n_phenos, Pheno, Pheno_t): 

    beta = - coef * np.mean(np.dot(inputs, inputs.T)) 
    grad, kernOut = kernel_gradient(kernel='sigmoid', inputs=inputs,
                                            gamma=coef, coef0=None,
                                            degree=None, gamma1=coef, delta1=beta, coef2=0.5 * np.var(inputs),
                                            coef3=1, adj_matrix=adj_matrix_kept)
        
    kernOut = kernOut + 1e-3 * np.eye(kernOut.shape[0])
    
    U, s, V = np.linalg.svd(grad.T)

    energy = np.cumsum(s**2) / np.sum(s**2)
    # Choose rank where 90% of the energy is retained
    threshold = 0.9
    rank = np.searchsorted(energy, threshold) + 1
    U = U[:, :rank]
    V = V[:, :rank]
    s = s[:rank]
    
    # objective
    ntps = inputs.shape[0]
    W = cp.Variable(shape=(inputs.shape[1], ntps))

    C = cp.Variable(shape=(n_phenos, ntps))

    K_tilde = U.conj().T @ W @ grad @ U

    constraints = [cp.norm(K_tilde, 2) - 1 <= 0.001] 
    outs = Pheno.T
    outs = outs[:, :ntps]
    outs_t = Pheno_t.T
    outs_t = outs_t[:, :ntps]
    
    alpha = 0.1; rho=0.99
    obj = cp.Minimize(cp.square(cp.norm(inputs_t.T - (W @ kernOut), "fro")) + alpha * (1 - rho) / 2 * cp.norm(W, "fro") + alpha * rho * cp.mixed_norm(W, 2, 1) +
                              cp.square(cp.norm(outs - (C @ kernOut), "fro")) + alpha * (1 - rho) / 2 * cp.norm(C, "fro") + alpha * rho * cp.mixed_norm(C, 2, 1))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver='CLARABEL', verbose=True)
    W = W.value
    
    # nonlinear part
    bias = np.mean(inputs, axis = 1)
            
    beta = - coef * np.mean(np.dot(bias[:, np.newaxis], bias[:, np.newaxis].T)) 
    _, kernOutbias = kernel_gradient(kernel="sigmoid", inputs=bias[:, np.newaxis], 
                                                gamma=coef, coef0=None,
                                                degree=None, gamma1=coef, delta1=beta, coef2=0.5 * np.var(bias[:, np.newaxis]),
                                                coef3=1, adj_matrix=adj_matrix_kept)
            
    kernOutbias = kernOutbias +  1e-3 * np.eye(kernOutbias.shape[0])
    # N is nonlinear residual term of Taylor series (full model - bias - linear dynamics)
    N = W.dot(kernOut + kernOutbias) - W.dot(kernOutbias) - np.linalg.multi_dot([W, grad, inputs.T.reshape(-1, inputs.T.shape[-1]),])
   # N = W1.dot(kernOut1) - 0 - np.linalg.multi_dot([W1, grad1, inputs1_keep.T.reshape(-1, inputs1_keep.T.shape[-1]),])
        # N = Bu
    u, s2, v = np.linalg.svd(N, full_matrices=False) # SVD of nonlinear forcing parts
    nonzero_inds = np.abs(s2) > 1e-16
    s2 = s2[nonzero_inds]
    sorted_inds = np.argsort(s2)[::-1]
    s2 = s2[sorted_inds]
    u = u[:, sorted_inds]

    N = N[sorted_inds, :]
        
    retained = np.argmax(np.cumsum(s2**2) / np.sum(s2**2) >=0.9) + 1 # choose based on 'energy' of singular values        

    u_manipulated = u[:, :retained] # B
    s2_man = s2[:retained]
    Vh_r = v[:retained, :]

    s2_man = np.maximum(s2[:retained], 1e-2)
    B = u_manipulated @ np.diag(s2_man) 
    
    return W @ grad, B, C.value @ grad
 
 
