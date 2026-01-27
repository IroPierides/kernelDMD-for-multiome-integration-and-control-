
import pandas as pd
import re
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import cvxpy as cp
import numpy as np
import os 
from sklearn.feature_selection import VarianceThreshold
from functools import reduce
import scipy 
from itertools import product
import random 
from matplotlib import cm
from matplotlib.lines import Line2D

np.random.seed(None)  # uncomment when you want to reproduce eigenmode results from one iteration -> always use seed 8
#os.environ["OMP_NUM_THREADS"] = "1" 

from preprocessing import *
from reconstructions import *
from SystemIdentification_and_Control import *
from figures import *

global MFsha_met, MFexp_met, Roseasha_met, Roseaexp_met

Roseasha_trans = pd.read_csv("Roseatran_c.tsv").iloc[:, 2:]
Roseasha_prot = pd.read_csv("Roseaprot_c.tsv").iloc[:, 2:]
Roseasha_met = pd.read_csv("Roseamet_c.csv").iloc[:, 2:]

MFsha_trans = pd.read_csv("Majortran_c.tsv").iloc[:, 2:]
MFsha_prot = pd.read_csv("Majorprot_c.tsv").iloc[:, 2:]
MFsha_met = pd.read_csv("Majormet_c.csv").iloc[:, 2:]

Roseaexp_trans = pd.read_csv("Roseatran_s.tsv").iloc[:, 2:]
Roseaexp_prot = pd.read_csv("Roseaprot_s.tsv").iloc[:, 2:]
Roseaexp_met = pd.read_csv("Roseamet_s.csv").iloc[:, 2:]

MFexp_trans = pd.read_csv("Majortran_s.tsv").iloc[:, 2:]
MFexp_prot = pd.read_csv("Majorprot_s.tsv").iloc[:, 2:]
MFexp_met = pd.read_csv("Majormet_s.csv").iloc[:, 2:]

MFsha_prot.columns = [re.sub("value.", "", x) for x in MFsha_prot.columns]
Roseasha_prot.columns = [re.sub("value.", "", x) for x in Roseasha_prot.columns]

MFsha_trans.columns = [re.sub("value.", "", x) for x in MFsha_trans.columns]
Roseasha_trans.columns = [re.sub("value.", "", x) for x in Roseasha_trans.columns]

MFsha_trans = MFsha_trans.loc[:, (MFsha_trans != 0).any(axis=0)]
Roseasha_trans = Roseasha_trans.loc[:, (Roseasha_trans != 0).any(axis=0)]

MFsha_prot = MFsha_prot.loc[:, (MFsha_prot != 0).any(axis=0)]
Roseasha_prot = Roseasha_prot.loc[:, (Roseasha_prot != 0).any(axis=0)]

MFexp_prot.columns = [re.sub("value.", "", x) for x in MFexp_prot.columns]
Roseaexp_prot.columns = [re.sub("value.", "", x) for x in Roseaexp_prot.columns]

MFexp_trans.columns = [re.sub("value.", "", x) for x in MFexp_trans.columns]
Roseaexp_trans.columns = [re.sub("value.", "", x) for x in Roseaexp_trans.columns]

MFexp_trans = MFexp_trans.loc[:, (MFexp_trans != 0).any(axis=0)]
Roseaexp_trans = Roseaexp_trans.loc[:, (Roseaexp_trans != 0).any(axis=0)]

MFexp_prot = MFexp_prot.loc[:, (MFexp_prot != 0).any(axis=0)]
Roseaexp_prot = Roseaexp_prot.loc[:, (Roseaexp_prot != 0).any(axis=0)]

sel = VarianceThreshold(threshold=0.3)
sel_var = sel.fit_transform(Roseasha_trans)
Roseasha_trans = Roseasha_trans[Roseasha_trans.columns[sel.get_support(indices=True)]]

sel = VarianceThreshold(threshold=0.3)
sel_var = sel.fit_transform(MFsha_trans)
MFsha_trans = MFsha_trans[MFsha_trans.columns[sel.get_support(indices=True)]]

sel = VarianceThreshold(threshold=0.3)
sel_var = sel.fit_transform(Roseaexp_trans)
Roseaexp_trans = Roseaexp_trans[Roseaexp_trans.columns[sel.get_support(indices=True)]]

sel = VarianceThreshold(threshold=0.3)
sel_var = sel.fit_transform(MFexp_trans)
MFexp_trans = MFexp_trans[MFexp_trans.columns[sel.get_support(indices=True)]]

# select only common columns between species

common_columns_trans = reduce(np.intersect1d, [
    Roseasha_trans.columns,
    MFsha_trans.columns,
    Roseaexp_trans.columns,
    MFexp_trans.columns
])

common_columns_prot = reduce(np.intersect1d, [
    Roseasha_prot.columns,
    MFsha_prot.columns,
    Roseaexp_prot.columns,
    MFexp_prot.columns
])

global Roseaexp_trans2, Roseasha_trans2, MFexp_trans2, MFsha_trans2, Roseaexp_prot2, Roseasha_prot2, MFexp_prot2, MFsha_prot2

Roseaexp_trans2 = Roseaexp_trans[common_columns_trans]
Roseasha_trans2 = Roseasha_trans[common_columns_trans]
MFexp_trans2 = MFexp_trans[common_columns_trans]
MFsha_trans2 = MFsha_trans[common_columns_trans]

Roseaexp_prot2 = Roseaexp_prot[common_columns_prot]
Roseasha_prot2 = Roseasha_prot[common_columns_prot]
MFexp_prot2 = MFexp_prot[common_columns_prot]
MFsha_prot2 = MFsha_prot[common_columns_prot]

# import cam features

met_names = pd.read_csv('met_names.csv').iloc[:, 0].values

cam = pd.read_csv("cam_features.tsv")
feature_labels = pd.read_csv("feature_table.csv")
feature_labels.replace("NA", np.nan, inplace=True)
feature_labels = feature_labels[~feature_labels["Name"].isna()].reset_index(drop=True)
feature_labels = feature_labels.dropna(subset=["Name", "Pathway"]).reset_index(drop=True)
feature_labels = feature_labels[feature_labels["Name"] != "NA"].reset_index(drop=True)

CAM_related_pathways2 = ['Carbon fixation by Calvin cycle',  'Malate_turnover', 'Carboxylation', 'Decarboxylation', 'Pyruvate metabolism', 'Glycolysis / Gluconeogenesis', 'Starch and sucrose metabolism',
'Circadian rhythm', 'Citrate cycle (TCA cycle)', 'Glyoxylate and dicarboxylate metabolism', 'Vacuolar_storage', 'Starch_syn/deg', 'Glycine, serine and threonine metabolism',
'Carbon_breakdown', 'Fructose and mannose metabolism','Galactose metabolism',  'Inositol phosphate metabolism', 'Regulation', 'Fatty acid syn/deg']

feature_labels = feature_labels[feature_labels['Pathway'].isin(CAM_related_pathways2)].reset_index(drop=True)

cam_tran_MF_sha, cam_prot_MF_sha = get_cam_features(MFsha_trans, MFsha_prot, cam['SingleCopyOG'].values)
cam_tran_Rosea_sha, cam_prot_Rosea_sha = get_cam_features(Roseasha_trans, Roseasha_prot, cam['SingleCopyOG'].values)

cam_tran_MF_exp, cam_prot_MF_exp = get_cam_features(MFexp_trans, MFexp_prot, cam['SingleCopyOG'].values)
cam_tran_Rosea_exp, cam_prot_Rosea_exp = get_cam_features(Roseaexp_trans, Roseaexp_prot, cam['SingleCopyOG'].values)

# 0-padding of missing cam_features

list1 = pd.Index(np.concatenate([cam_tran_Rosea_sha.columns, cam_tran_Rosea_exp.columns, cam_tran_MF_exp.columns])).difference(cam_tran_MF_sha.columns)
list2 = pd.Index(np.concatenate([cam_tran_MF_sha.columns, cam_tran_MF_exp.columns, cam_tran_Rosea_exp.columns])).difference(cam_tran_Rosea_sha.columns)
list3 = pd.Index(np.concatenate([cam_prot_Rosea_sha.columns, cam_prot_Rosea_exp.columns, cam_prot_MF_exp.columns])).difference(cam_prot_MF_sha.columns)
list4 = pd.Index(np.concatenate([cam_prot_MF_sha.columns, cam_prot_MF_exp.columns, cam_prot_Rosea_exp.columns])).difference(cam_prot_Rosea_sha.columns)

list1_1 = pd.Index(np.concatenate([cam_tran_Rosea_sha.columns, cam_tran_Rosea_exp.columns, cam_tran_MF_sha.columns])).difference(cam_tran_MF_exp.columns)
list2_1 = pd.Index(np.concatenate([cam_tran_MF_sha.columns, cam_tran_MF_exp.columns, cam_tran_Rosea_sha.columns])).difference(cam_tran_Rosea_exp.columns)
list3_1 = pd.Index(np.concatenate([cam_prot_Rosea_sha.columns, cam_prot_Rosea_exp.columns, cam_prot_MF_sha.columns])).difference(cam_prot_MF_exp.columns)
list4_1 = pd.Index(np.concatenate([cam_prot_MF_sha.columns, cam_prot_MF_exp.columns, cam_prot_Rosea_sha.columns])).difference(cam_prot_Rosea_exp.columns)

cam_tran_MF_sha = add_missing_columns(cam_tran_MF_sha, list1)
cam_prot_MF_sha = add_missing_columns(cam_prot_MF_sha, list3)
cam_tran_Rosea_sha = add_missing_columns(cam_tran_Rosea_sha, list2)
cam_prot_Rosea_sha = add_missing_columns(cam_prot_Rosea_sha, list4)

cam_tran_MF_exp = add_missing_columns(cam_tran_MF_exp, list1_1)
cam_prot_MF_exp = add_missing_columns(cam_prot_MF_exp, list3_1)
cam_tran_Rosea_exp = add_missing_columns(cam_tran_Rosea_exp, list2_1)
cam_prot_Rosea_exp = add_missing_columns(cam_prot_Rosea_exp, list4_1)

# phenotypes

# major

Malate_MF_sha = MFsha_met.iloc[:, pd.DataFrame(np.where(MFsha_met.columns.isin(['Malic_acid']))).T[0].values.tolist()]
Malate_MF_exp = MFexp_met.iloc[:, pd.DataFrame(np.where(MFexp_met.columns.isin(['Malic_acid']))).T[0].values.tolist()]

MFgas = pd.read_csv("MFgas.csv")

MFsha_prot.columns = [re.sub("values.", "", x) for x in MFsha_prot.columns]
MFexp_prot.columns = [re.sub("values.", "", x) for x in MFexp_prot.columns]

CAM_pheno_M_sha = MFsha_prot.iloc[:, pd.DataFrame(np.where(MFsha_prot.columns.isin(['OG0002610::H1', 'OG0001285::H1', 'OG0004358::H2', 'OG0001386::H1']))).T[0].values.tolist()]  # 'OG0001285::H1' PPC1, 'OG0004358::H2' PHO1, 'OG0001386::H1' PPD

CAM_pheno_M_exp = MFexp_prot.iloc[:, pd.DataFrame(np.where(MFexp_prot.columns.isin(['OG0002610::H1', 'OG0001285::H1', 'OG0004358::H2', 'OG0001386::H1']))).T[0].values.tolist()]  # 'OG0001285::H1' PPC1,'OG0004358::H2' PHO1, 'OG0001386::H1' PPD

# rosea

Malate_Rosea_sha = Roseasha_met.iloc[:, pd.DataFrame(np.where(Roseasha_met.columns.isin(['Malic_acid']))).T[0].values.tolist()]
Malate_Rosea_exp = Roseaexp_met.iloc[:, pd.DataFrame(np.where(Roseaexp_met.columns.isin(['Malic_acid']))).T[0].values.tolist()]

Roseagas = pd.read_csv("Roseagas.csv")

Roseasha_prot.columns = [re.sub("values.", "", x) for x in Roseasha_prot.columns]
Roseaexp_prot.columns = [re.sub("values.", "", x) for x in Roseaexp_prot.columns]

CAM_pheno_R_sha = Roseasha_prot.iloc[:, pd.DataFrame(np.where(Roseasha_prot.columns.isin(['OG0002610::H1', 'OG0001285::H1', 'OG0004358::H2', 'OG0001386::H1']))).T[0].values.tolist()]  # PEPC1, PCK1, PHO1, PPD

CAM_pheno_R_exp = Roseaexp_prot.iloc[:, pd.DataFrame(np.where(Roseaexp_prot.columns.isin(['OG0002610::H1', 'OG0001285::H1', 'OG0004358::H2', 'OG0001386::H1']))).T[0].values.tolist()]  # PEPC1, PCK1, PHO1, PPD

def process_kernel_grid_item(i, j, val, grid_row, species, features, names, products, features2):
        # Kernel computation logic for a single (i, val, grid_row) combination
        random.seed(None) #uncomment for replication of one iteration results seen in the paper 
        np.random.seed(None)
        kernel_outputs = {"params": [], "K_tilde": [], "eVals": [], "modes": [], "mode pairs": [],
                        "mode_amplitudes": [], "gradients": [], "Q": [], "R": [], "U": [], "C": [], "W": [],
                        "V": [], "B": [], "C_full": [],"C_exo": [],"inputs": [], "exo_inputs": [], "man_inputs": [], "man_inputs_index": [], "V_man": [], "V_exo": [], 
                        "spectral_cluster": [], "data_error": [], "W_man": [], "W_exo": [], "controllable": [], 
                        "output_error": [], "features": [], "names": [], "condition number": [], "r2_data_no_c": [], "inputs_t": [],
                        "s": [], "eigs_in": [], "r2_data_full_model": [], "cd_outs_full_model": [], "adj_matrix": [],  
                        "K_full": [], "K_exo": [], "recon": [], "recon_modal": [], "r2_modal": [], "CO2_ind": [], "obs_rank": [], "modal_dictionary": [],
                        "cd_outs": [], "r2_data": [], "n_modes": [], "coef2": [], "beta": []}
        
        param = grid_row
        inputs = np.concatenate((val[0], val[1], val[2]), axis=1)

        inputs_t = np.concatenate((val[6], val[7], val[8]), axis=1)
        ntps = inputs.shape[0]
        np.set_printoptions(precision=16)
        
        # Kernel training
        beta = - param[4] * np.mean(np.dot(inputs, inputs.T))
        coef2 = 0.1 * np.var(inputs)
        adj_matrix = make_int_matrix(names)

        grad1, kernOut1 = kernel_gradient(
                kernel=param[0], inputs=inputs,
                gamma=param[1], coef0=param[2],
                degree=param[3], gamma1=param[4],
                delta1=beta, coef2=coef2,
                coef3=param[7], adj_matrix=adj_matrix
            )
            
        kernOut1 += 1e-3 * np.eye(kernOut1.shape[0])
        U, s, V = scipy.linalg.svd(grad1.T, lapack_driver='gesvd')
        energy = np.cumsum(s**2) / np.sum(s**2)

        # Choose rank where 99% of the energy is retained
        threshold = 0.99
        rank = np.searchsorted(energy, threshold) + 1
        
        U = U[:, :rank]
        V = V.conj().T[:, :rank]
        s = s[:rank]
        
        W1 = cp.Variable((inputs.shape[1], ntps))

        C = cp.Variable((n_phenos, ntps))
        K_tilde = U.T @ W1 @ grad1 @ U
        constraints = [cp.norm(K_tilde, 2) - 1 <= 0.001]

        outs = val[3].T
        outs_t = val[4].T

        alpha = 1
        obj = cp.Minimize(
                cp.square(cp.norm(inputs_t.T - W1 @ kernOut1, "fro")) +
                alpha * (1 - 0.99 ) * cp.norm(W1, "fro") + alpha * 0.99 * cp.mixed_norm(W1, 2, 1) +
                cp.square(cp.norm(outs - C @ kernOut1, "fro")) +
                alpha * (1 - 0.99 ) * cp.norm(C, "fro") + alpha * 0.99 * cp.mixed_norm(C, 2, 1)
            )
        prob = cp.Problem(obj)
        prob.solve(solver='CLARABEL')
        W1 = W1.value
        C = C.value
    
        C_full = C @ grad1
        K_full = W1 @ grad1 
        W = W1
                
        # nonlinear operator
        inputs2 = inputs.copy()
        inputs2_t = inputs_t.copy()
            
        # bias is base state 
        bias = np.mean(inputs2, axis = 1)
            
        beta = - param[4] *  np.mean(np.dot(bias[:, np.newaxis], bias[:, np.newaxis].T)) 
        coef2 = 0.1 * np.var(inputs)
        _, kernOutbias = kernel_gradient(kernel=param[0], inputs=bias[:, np.newaxis], 
                                                gamma=1e-3, coef0=param[2],
                                                degree=param[3], gamma1=param[4], delta1=beta, coef2=0.1 * np.var(bias[:, np.newaxis]),
                                                coef3=param[7], adj_matrix=adj_matrix)
            
        kernOutbias = kernOutbias + 1e-3 * np.eye(kernOutbias.shape[0])

        fluctuations = inputs - bias[:, np.newaxis]
        # N is nonlinear residual term of Taylor series (full model - bias - linear dynamics)
        N = W.dot(kernOut1 + kernOutbias) - W.dot(kernOutbias) - np.linalg.multi_dot([W, grad1, inputs.T.reshape(-1, inputs.T.shape[-1]),])
       # N = W.dot(kernOut1) - 0 - np.linalg.multi_dot([W, grad1, inputs.T.reshape(-1, inputs.T.shape[-1]),])
        # N = Bu
        u, s2, v = np.linalg.svd(N, full_matrices=False) # SVD of nonlinear forcing parts

        nonzero_inds = np.abs(s2) > 1e-16
        s2 = s2[nonzero_inds]

        sorted_inds = np.argsort(s2)[::-1]
        s2 = s2[sorted_inds]
        u = u[:, sorted_inds]
        N = N[sorted_inds, :]
        
        # keep only high singular values for nonlinear parts
        retained = np.argmax(np.cumsum(s2**2) / np.sum(s2**2) >= 0.9) + 1 # choose based on 'energy' of singular values        

        u_manipulated = u[:, :retained] # B
        s2_man = s2[:retained]
        Vh_r = v[:retained, :]
        u_exogenous = u[:, retained:]

        exogenous_inputs = inputs 
        s2_man = np.maximum(s2[:retained], 1e-2) 
        B = u_manipulated @ np.diag(s2_man)        
        manipulated_inputs = Vh_r 
        manipulated_inputs = manipulated_inputs.T

        K_tilde = K_tilde.value 

        # eigendecomposition       
        eigenvalues, eigenvectors = np.linalg.eig(K_tilde)
        nonzero_inds = np.abs(eigenvalues) > 1e-6
        eigenvalues = eigenvalues[nonzero_inds]
        eigenvectors = eigenvectors[:, nonzero_inds]

        # Sort eigenvalues, descending based on modulus.
        sorted_inds = np.argsort(-np.real(eigenvalues))
        eVals_r = eigenvalues[sorted_inds]
        eVecs_r = eigenvectors[:, sorted_inds]

        # eigenmodes      
        Phi = np.linalg.multi_dot([W1, V, np.diag(s), eVecs_r, np.diag(1 / eVals_r), ])
        Phi = Phi / np.linalg.norm(Phi, axis=0)   
        
        mode_pairs = []
        cnt = 0
        while cnt in range(len(eVals_r)):
            if cnt + 1 != len(eVals_r):
                if np.abs(eVals_r[cnt]) == np.abs(eVals_r[cnt + 1]):
                    mode_pairs.append([cnt, cnt + 1])
                    cnt += 2
                else:
                    mode_pairs.append([cnt])
                    cnt += 1
            else:
                mode_pairs.append([cnt])
                cnt += 1
        
        clean_name = re.sub(r'[^\w\s-]', '', list(species)[j])  
        clean_name = clean_name.replace(" ", "_")
        
        kmax = 90 
        kmax_inds_list = []
        for ii in range(Phi.shape[1]):
            ind = Phi[:, ii].argsort()[-kmax:]
            kmax_inds_list.append(ind[:30])
         #   mask = [i for i in range(Phi.shape[0]) if i not in ind]
         #   Phi[mask, ii] = 0
                
        max_norm = 60
       #b_r1 = np.linalg.pinv(Phi) @ fluctuations.T
        b_r1 = cp.Variable((Phi.shape[1], inputs.shape[0]))
        constraints = [cp.norm(b_r1, 2) <= max_norm]
        obj1 = cp.Minimize(cp.norm(outs_t - C_full @ Phi @ np.diag(eVals_r) @ b_r1, "fro")) #+ 0.01 * cp.norm(b_r1, 1))
        prob1 = cp.Problem(obj1, constraints)
        prob1.solve(solver='CLARABEL')
        b_r1 = b_r1.value
        
        # reconstructions 
        recon_modal, cd_modal, r2_modal, modal_error = modal_reconstruction(C=C_full, outs=outs, eVals_r=eVals_r, modes=Phi, b_r0=b_r1, ntimepts=12, nmodes=len(mode_pairs), n_phenos=n_phenos, mode_pairs=mode_pairs)
        CO2_ind = []
        for j in range(len(r2_modal)):
            if r2_modal[j][2] > 0.3:
                CO2_ind.append(j)
                
        recon, cd_outs, mse1, r2_outs, error_out = reconstructed_outputs(K=K_full, C=C_full, outs=val[3].T, inputs=inputs, ntimepts=12, n_phenos=n_phenos)
        
        snapshots, cd_data, mse, r2_data = reconstructed_data(K=K_full, inputs1=inputs, ntimepts=21)      

        # obserability and network clustering                      
        if all(np.abs(eVals_r) <= 1):
            eigs_in = True
        else:
            eigs_in = False               
        
        '''
        # network file
        df = pd.DataFrame(K_full)
        df.index.name = 'Target'
        edges = df.reset_index().melt('Target', value_name='Weight', var_name='Source').query('Source != Target')
        edges['Weight'] = np.real(edges['Weight'])
        edges['Source_names'] = np.asarray(names)[edges['Source'].values.tolist()]
        edges['Source_Prod'] = np.asarray(products)[edges['Source'].values.tolist()]
        edges['Target_names'] = np.asarray(names)[edges['Target'].values.tolist()]
        edges['Target_Prod'] = np.asarray(products)[edges['Target'].values.tolist()]
        edges = edges.sort_values(by='Weight', ascending=False).iloc[:500, :]

        edges.to_csv(f'{clean_name}_sigmoid.csv')
        '''
        
        ## leave-3-out cross validation 
        
        #  cross_validation(inputs=inputs, inputs_t=inputs_t, n_phenos=6, rank=rank, Pheno=outs, param0=param[0], param1=param[0], param2=param[2], param3=param[3], param4=param[4], param7=param[7])
        
        kernel_outputs["params"].append(param)
        # eigen outputs
        kernel_outputs["eVals"].append(eVals_r)
        kernel_outputs["modes"].append(Phi)
        kernel_outputs["mode pairs"].append(mode_pairs)
        kernel_outputs["mode_amplitudes"].append(b_r1)
        # decomposed outputs
        kernel_outputs["gradients"].append(grad1)
        kernel_outputs["U"].append(U)
        kernel_outputs["V"].append(V)
        kernel_outputs["s"].append(s)
        # operators
        kernel_outputs["C_full"].append(C_full)
        kernel_outputs["K_full"].append(K_full)
        kernel_outputs["B"].append(B)
        # accuracies
        kernel_outputs["recon"].append(recon)
        kernel_outputs["recon_modal"].append(recon_modal)
        kernel_outputs["r2_modal"].append(r2_modal)
        kernel_outputs["cd_outs"].append(r2_outs)
        kernel_outputs["r2_data"].append(r2_data) #{"R\u00b2": r2_data, "corr_coef": cd_data, "mse": mse})
        kernel_outputs["CO2_ind"].append(CO2_ind) #{"R\u00b2": r2_data, "corr_coef": cd_data, "mse": mse})

        # feature types
        kernel_outputs["n_modes"].append(rank)
        kernel_outputs["eigs_in"].append(eigs_in)
        kernel_outputs["inputs"].append(inputs)
        kernel_outputs["inputs_t"].append(inputs_t)
        kernel_outputs["man_inputs"].append(manipulated_inputs)
        kernel_outputs["V_exo"].append(u_exogenous)
        kernel_outputs["V_man"].append(u_manipulated)        
        kernel_outputs["coef2"].append(coef2)
        kernel_outputs["beta"].append(beta)
        kernel_outputs["adj_matrix"].append(adj_matrix)
        
        return kernel_outputs
    
def eval_kernel(input_X1, input_X2, n_modes, input_X3, Pheno, Pheno_t, outputs, input_Y1, input_Y2, input_Y3, input_df1,
                input_df2, input_df3, comparison, iter, products, names, features, features2):
    global n_phenos
    global rank
    rank = n_modes
    n_phenos = Pheno[0].shape[1]
    np.random.seed(None)
    random.seed(None)
    param_linear = {'kernel': ["linear"], 'gamma': [0.0001], 'coef': [1], 'degree': [2],  'gamma1': [0.0001], 'delta1': [3], 'coef2': [0.1], 'coef3': [1], 'l': [3], 'retained': [3]}

    param_rbf = {'kernel': ["rbf"], 'gamma': [0.001], 'coef': [1], 'degree': [2], 'gamma1': [0.0001], 'delta1': [3], 'coef2': [0.1], 'coef3': [1], 'l': [3], 'retained': [3]}

    param_poly = {'kernel': ["poly"], 'gamma': [0.001], 'coef': [1], 'degree': [5], 'gamma1': [0.0001], 'delta1': [3], 'coef2': [0.1], 'coef3': [1], 'l': [3], 'retained': [3]}
    
    param_sigmoid = {'kernel': ["sigmoid"], 'gamma': [0.0001], 'coef': [1], 'degree': [3], 'gamma1': [0.001], 'delta1': [2], 'coef2': [0.1], 'coef3': [3], 'l': [3], 'retained': [3]}

    param_grid3 = {
        r'$\it{C.\ major}$ (control)': [input_X1[0], input_X2[0], input_X3[0], Pheno[0], Pheno_t[0], outputs[0], input_Y1[0], input_Y2[0], input_Y3[0], input_df1[0], input_df2[0], input_df3[0]],
        r'$\it{C.\ major}$ (stress)': [input_X1[1], input_X2[1], input_X3[1], Pheno[1], Pheno_t[1], outputs[1], input_Y1[1],  input_Y2[1], input_Y3[1], input_df1[1], input_df2[1], input_df3[1]],
        r'$\it{C.\ rosea}$ (control)': [input_X1[2], input_X2[2], input_X3[2], Pheno[2], Pheno_t[2], outputs[2], input_Y1[2], input_Y2[2], input_Y3[2], input_df1[2], input_df2[2], input_df3[2]],
        r'$\it{C.\ rosea}$ (stress)': [input_X1[3], input_X2[3], input_X3[3], Pheno[3], Pheno_t[3], outputs[3], input_Y1[3], input_Y2[3], input_Y3[3], input_df1[3], input_df2[3], input_df3[3]]}
    
    grid0 = pd.DataFrame(product(*param_linear.values()), columns=list(param_linear.keys()))
    grid1 = pd.DataFrame(product(*param_rbf.values()), columns=list(param_rbf.keys()))
    grid2 = pd.DataFrame(product(*param_poly.values()), columns=list(param_poly.keys()))
    grid3 = pd.DataFrame(product(*param_sigmoid.values()), columns=list(param_sigmoid.keys()))

    grid = grid3 # for computational efficiency try each kernel separately; or try pd.concat((grid0, grid1, grid2, grid3), axis=0)
    grid = grid.reset_index(drop=True)

    species_kernels = {"species": [], "kernel_outputs": []}

    r2_data_cols = [[] for _ in range(4)]
    cd_outs_cols = [[] for _ in range(4)]
    condition_num_cols = [[] for _ in range(4)]
    modal_dictionaries = [[] for _ in range(4)]
    
    for i in range(grid.shape[0]):
        for j, val in enumerate(param_grid3.values()):
            kernel_outs = process_kernel_grid_item(
                i, j, val, grid.iloc[i, :], param_grid3.keys(), features, names, products, features2
            )

            # Append metrics to the correct list per species-condition
            r2_data_cols[j].append(kernel_outs["r2_data"])
            modal_dictionaries[j].append(kernel_outs["modal_dictionary"])
            cd_outs_cols[j].append(kernel_outs["cd_outs"])
            species_kernels["kernel_outputs"].append(kernel_outs)
    
    species_kernels["species"].append(list(param_grid3.keys()))

    grid['r2_data multi (control)'] = r2_data_cols[0]
    grid['r2_data multi (stress)'] = r2_data_cols[1]
    grid['r2_data rosea (control)'] = r2_data_cols[2]
    grid['r2_data rosea (stress)'] = r2_data_cols[3]

    grid['Accuracy of outputs multi (control)'] = cd_outs_cols[0]
    grid['Accuracy of outputs multi (stress)'] = cd_outs_cols[1]
    grid['Accuracy of outputs rosea (control)'] = cd_outs_cols[2]
    grid['Accuracy of outputs rosea (stress)'] = cd_outs_cols[3]
    
    grid['eigs in multi (control)'] = species_kernels['kernel_outputs'][0]["eigs_in"][0]
    grid['eigs in multi (stress)'] = species_kernels['kernel_outputs'][1]["eigs_in"][0]
    grid['eigs in rosea (control)'] = species_kernels['kernel_outputs'][2]["eigs_in"][0]
    grid['eigs in rosea (stress)'] = species_kernels['kernel_outputs'][3]["eigs_in"][0]

    # grid.to_csv(f"all_kernel_evals.csv")    # save reconstruction accuracies  
    
    # mode clusters figure (eigenvalues and mode amplitudes)
    col_ind1, col_ind2 = mode_clusters_fig(eVals_1=species_kernels['kernel_outputs'][0]["eVals"][0],
                        eVals_4=species_kernels['kernel_outputs'][2]["eVals"][0],
                        mode_pairs1=species_kernels['kernel_outputs'][0]["mode pairs"][0],
                        mode_pairs4=species_kernels['kernel_outputs'][2]["mode pairs"][0],
                        b_1=species_kernels['kernel_outputs'][0]["mode_amplitudes"][0],
                        b_4=species_kernels['kernel_outputs'][2]["mode_amplitudes"][0],
                        label1=species_kernels["species"][0][0],
                        label4=species_kernels["species"][0][2],
                        figlabel="sigmoid")
    
    # phenotype reconstruction figure 
    CO1_ind, CO2_ind = pheno_recons_fig(label1=species_kernels["species"][0][0], label2=species_kernels["species"][0][2], 
                    Pheno1=Pheno[0], Pheno2=Pheno[2], 
                    recon1=species_kernels['kernel_outputs'][0]["recon"][0], recon2=species_kernels['kernel_outputs'][2]["recon"][0], 
                    cd1=species_kernels['kernel_outputs'][0]["cd_outs"][0], cd2=species_kernels['kernel_outputs'][2]["cd_outs"][0], 
                    recon_modal1=species_kernels['kernel_outputs'][0]["recon_modal"][0], recon_modal2=species_kernels['kernel_outputs'][2]["recon_modal"][0],
                    cd_modes1=species_kernels['kernel_outputs'][0]["r2_modal"][0], cd_modes2=species_kernels['kernel_outputs'][2]["r2_modal"][0], col_ind1=col_ind1, col_ind2=col_ind2)
    
    
    # eigenmode consensus network
    G1, G2 = modes_heatmap(Phi1= species_kernels['kernel_outputs'][0]["modes"][0],
                mode_pairs1=species_kernels['kernel_outputs'][0 ]["mode pairs"][0],
                Phi4=species_kernels['kernel_outputs'][2]["modes"][0],
                evals1 = species_kernels['kernel_outputs'][0]["eVals"][0], 
                evals4 = species_kernels['kernel_outputs'][2]["eVals"][0], 
                mode_pairs4=species_kernels['kernel_outputs'][2]["mode pairs"][0],
                names=features2, adj_matrix=species_kernels['kernel_outputs'][2]["adj_matrix"][0], products=products, col_ind1=col_ind1, col_ind2=col_ind2, 
                CO2_ind1=CO1_ind, CO2_ind2=CO2_ind)
    
    # PART 2: SYSTEM CONTROL 

    # System reduction via HSV ranking 
    
    important_states1, hsv1 = gramian_hsv_ranking(species_kernels['kernel_outputs'][0]["K_full"][0], species_kernels['kernel_outputs'][0]["B"][0], species_kernels['kernel_outputs'][0]["C_full"][0] , k=50)
    important_states2, hsv2 = gramian_hsv_ranking(species_kernels['kernel_outputs'][2]["K_full"][0], species_kernels['kernel_outputs'][2]["B"][0] , species_kernels['kernel_outputs'][2]["C_full"][0] , k=50)

    names = np.asarray(names)
    features = np.asarray(features)
    features2 = np.asarray(features2)
    keep_states = np.unique(np.concatenate([important_states1, important_states2]))
    
    red_feat_space1 = pd.DataFrame(features2[important_states1])
    red_feat_space1.columns = ["Feature"]
    red_feat_space1['hsv_value'] = hsv1
    
    red_feat_space2 = pd.DataFrame(features2[important_states2])
    red_feat_space2.columns = ["Feature"]
    red_feat_space2['hsv_value'] = hsv2

    names_kept = names[keep_states]
    # System reidentification in reduced feature space
    K_A, B1, C_A = system_reIdendification(inputs=species_kernels['kernel_outputs'][0]["inputs"][0] [:, keep_states], inputs_t=species_kernels['kernel_outputs'][0]["inputs_t"][0][:, keep_states], coef=0.001, adj_matrix_kept=make_int_matrix(names_kept), n_phenos=Pheno[0].shape[1], Pheno=Pheno[0], Pheno_t=Pheno_t[0])
    K_B, B2, C_B = system_reIdendification(inputs=species_kernels['kernel_outputs'][2]["inputs"][0] [:, keep_states], inputs_t=species_kernels['kernel_outputs'][2]["inputs_t"][0][:, keep_states], coef=0.001, adj_matrix_kept=make_int_matrix(names_kept), n_phenos=Pheno[3].shape[1], Pheno=Pheno[3], Pheno_t=Pheno_t[3])

    # system alignment and projection via matrix T
    
    T = cp.Variable((K_A.shape[1], K_A.shape[0]))

    state_error = 0
    
    state_error = cp.square(cp.norm(C_B @ K_B - C_A @ K_A @ T, "fro"))

    reg_term = cp.norm(T, "fro") ** 2
    obj = cp.Minimize(state_error + 0.001 * reg_term)
    prob = cp.Problem(obj)
    prob.solve()
    T = T.value
    
    # linear Model Predictive Control 

    x_inputs, u_inputs, control_sensitivity, cd = linearMPC(B1, C_A, K_A @ T, species_kernels['kernel_outputs'][0]["inputs"][0] [:, keep_states], species_kernels['kernel_outputs'][2]["inputs"][0] [:, keep_states], 
                                                            Pheno1=Pheno[0], Pheno2=Pheno[3], n_phenos=n_phenos, n_controls=B1.shape[1], features_kept=names_kept, with_fig=False)
    
   
    return grid, control_sensitivity, G1, G2, cd, red_feat_space1, red_feat_space2

def run_iteration(h, labels):
        # Generate data with added noise using the grouped data
        np.random.seed(None) # set a constant number in every seed for reproducible results 
        random.seed(None)
        global cam_tran_MF_sha, cam_tran_MF_exp, cam_prot_MF_sha, cam_prot_MF_exp, cam_tran_Rosea_sha, cam_tran_Rosea_exp, cam_prot_Rosea_sha, cam_prot_Rosea_exp
        global Roseaexp_trans2, Roseasha_trans2, MFexp_trans2, MFsha_trans2, Roseaexp_prot2, Roseasha_prot2, MFexp_prot2, MFsha_prot2
        global MFsha_met, MFexp_met, Roseasha_met, Roseaexp_met

        MFsha_trans2 = MFsha_trans2.iloc[:, MFsha_trans2.columns.isin(labels["SingleCopyOG"])]        
        MFsha_prot2 = MFsha_prot2.iloc[:, MFsha_prot2.columns.isin(labels["SingleCopyOG"])]        
        Roseasha_trans2 = Roseasha_trans2.iloc[:, Roseasha_trans2.columns.isin(labels["SingleCopyOG"])]        
        Roseasha_prot2 = Roseasha_prot2.iloc[:, Roseasha_prot2.columns.isin(labels["SingleCopyOG"])]        
        MFexp_trans2 = MFexp_trans2.iloc[:, MFexp_trans2.columns.isin(labels["SingleCopyOG"])]        
        MFexp_prot2 = MFexp_prot2.iloc[:, MFexp_prot2.columns.isin(labels["SingleCopyOG"])]        
        Roseaexp_trans2 = Roseaexp_trans2.iloc[:, Roseaexp_trans2.columns.isin(labels["SingleCopyOG"])]        
        Roseaexp_prot2 = Roseaexp_prot2.iloc[:, Roseaexp_prot2.columns.isin(labels["SingleCopyOG"])]        #

        transcriptome_concat_MF_sha = scale_and_transform(MFsha_trans2, scaler2)
        proteome_concat_MF_sha = scale_and_transform(MFsha_prot2, scaler2)
        transcriptome_concat_R_sha = scale_and_transform(Roseasha_trans2, scaler2)
        proteome_concat_R_sha = scale_and_transform(Roseasha_prot2, scaler2)
        transcriptome_concat_MF_exp = scale_and_transform(MFexp_trans2, scaler2)
        proteome_concat_MF_exp = scale_and_transform(MFexp_prot2, scaler2)
        transcriptome_concat_R_exp = scale_and_transform(Roseaexp_trans2, scaler2)
        proteome_concat_R_exp = scale_and_transform(Roseaexp_prot2, scaler2)

        # Feature selection using PCA
        most_important1_MF_sha = get_important_features(transcriptome_concat_MF_sha)
        most_important2_MF_sha = get_important_features(proteome_concat_MF_sha)
        most_important1_MF_exp = get_important_features(transcriptome_concat_MF_exp)
        most_important2_MF_exp = get_important_features(proteome_concat_MF_exp)
        most_important1_R_sha = get_important_features(transcriptome_concat_R_sha)
        most_important2_R_sha = get_important_features(proteome_concat_R_sha)
        most_important1_R_exp = get_important_features(transcriptome_concat_R_exp)
        most_important2_R_exp = get_important_features(proteome_concat_R_exp)

        # Consolidating important features
        important_1 = list(set(most_important1_MF_sha + most_important1_R_sha + most_important1_R_exp + most_important1_MF_exp))
        important_2 = list(set(most_important2_MF_sha + most_important2_R_sha + most_important2_MF_exp + most_important2_R_exp))

        # Subsetting DataFrames based on important features
        dataframes = {
            'Roseasha_trans2': Roseasha_trans2,
            'MFsha_trans2': MFsha_trans2,
            'Roseasha_prot2': Roseasha_prot2,
            'MFsha_prot2': MFsha_prot2,
            'Roseaexp_trans2': Roseaexp_trans2,
            'MFexp_trans2': MFexp_trans2,
            'Roseaexp_prot2': Roseaexp_prot2,
            'MFexp_prot2': MFexp_prot2
        }

        for name, df in dataframes.items():
            if 'trans' in name:
                dataframes[name] = df.iloc[:, important_1]
            else:
                dataframes[name] = df.iloc[:, important_2]

        Roseasha_trans2 = dataframes['Roseasha_trans2']
        MFsha_trans2 = dataframes['MFsha_trans2']
        Roseasha_prot2 = dataframes['Roseasha_prot2']
        MFsha_prot2 = dataframes['MFsha_prot2']
        Roseaexp_trans2 = dataframes['Roseaexp_trans2']
        MFexp_trans2 = dataframes['MFexp_trans2']
        Roseaexp_prot2 = dataframes['Roseaexp_prot2']
        MFexp_prot2 = dataframes['MFexp_prot2']

        scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
        Roseasha_trans3 = sort_and_concat(Roseasha_trans2, cam_tran_Rosea_sha)
        MFsha_trans3 = sort_and_concat(MFsha_trans2, cam_tran_MF_sha)
        Roseasha_prot3 = sort_and_concat(Roseasha_prot2, cam_prot_Rosea_sha)
        MFsha_prot3 = sort_and_concat(MFsha_prot2, cam_prot_MF_sha)

        Roseaexp_trans3 = sort_and_concat(Roseaexp_trans2, cam_tran_Rosea_exp)
        MFexp_trans3 = sort_and_concat(MFexp_trans2, cam_tran_MF_exp)
        Roseaexp_prot3 = sort_and_concat(Roseaexp_prot2, cam_prot_Rosea_exp)
        MFexp_prot3 = sort_and_concat(MFexp_prot2, cam_prot_MF_exp)

        MFsha_met = MFsha_met.sort_index(axis=1)
        Roseasha_met = Roseasha_met.sort_index(axis=1)

        MFexp_met = MFexp_met.sort_index(axis=1)
        Roseaexp_met = Roseaexp_met.sort_index(axis=1)
        
        features_tran = list(Roseasha_trans3.iloc[:12, :].columns)
        features_prot = list(Roseasha_prot3.iloc[:12, :].columns)
        features_met = list(met_names)
        features = features_tran + features_prot + features_met
        features2 = [i + "_transcript" for i in features_tran] + [i + "_protein" for i in features_prot] + features_met

        names = list(np.repeat("NA", len(features)))
        names2 = list(np.repeat("NA", len(features)))

        products = list(np.repeat("NA", len(features)))
        for k2, val1 in enumerate(features):
            for j2, val2 in enumerate(feature_labels['SingleCopyOG']):
                if val2 == val1:
                    names[k2] = str(feature_labels['Name'].iloc[j2]) + ' (' + features2[k2] + ')'
                    names2[k2] = str(feature_labels['Name'].iloc[j2]) 
                    products[k2] = str(feature_labels['Product'].iloc[j2]) + ' (' + features2[k2] + ')'
                elif val1 in features_met:
                    names[k2] = val1
                    names2[k2] = val1
                    products[k2] = val1

        products = [re.sub("%2C", "", x) for x in products]
        names = [re.sub("%2C", "", x) for x in names]
        names2 = [re.sub("%2C", "", x) for x in names2]
    
        timepoint_data = [
        (MFsha_trans3, MFsha_prot3, MFsha_met, Malate_MF_sha, CAM_pheno_M_sha, MFgas),
        (MFexp_trans3, MFexp_prot3, MFexp_met, Malate_MF_exp, CAM_pheno_M_exp, MFgas),
        (Roseasha_trans3, Roseasha_prot3, Roseasha_met, Malate_Rosea_sha, CAM_pheno_R_sha, Roseagas),
        (Roseaexp_trans3, Roseaexp_prot3, Roseaexp_met, Malate_Rosea_exp, CAM_pheno_R_exp, Roseagas) ]
        
        data_sets = [timepoint_rearrangements(data1=d[0], data2=d[1], data3=d[2], output1=d[3], output2=d[4], output3=d[5]) for d in tuple(timepoint_data)]

        # Unpack the results
        (train1_X_M_sha, train2_X_M_sha, train3_X_M_sha, train1_Y_M_sha, train2_Y_M_sha, train3_Y_M_sha, Pheno_t1_Multi_sha, Pheno_t0_Multi_sha), \
        (train1_X_M_exp, train2_X_M_exp, train3_X_M_exp, train1_Y_M_exp, train2_Y_M_exp, train3_Y_M_exp, Pheno_t1_Multi_exp, Pheno_t0_Multi_exp), \
        (train1_X_R_sha, train2_X_R_sha, train3_X_R_sha, train1_Y_R_sha, train2_Y_R_sha, train3_Y_R_sha, Pheno_t1_Rosea_sha, Pheno_t0_Rosea_sha), \
        (train1_X_R_exp, train2_X_R_exp, train3_X_R_exp, train1_Y_R_exp, train2_Y_R_exp, train3_Y_R_exp, Pheno_t1_Rosea_exp, Pheno_t0_Rosea_exp) = data_sets

        # Evaluate kernel with noisy data
        grid, control_sensitivity, G1, G2, cd, red_feat_space1, red_feat_space2 = eval_kernel(
            input_X1=[train1_X_M_sha, train1_X_M_exp, train1_X_R_sha, train1_X_R_exp],
            input_X2=[train2_X_M_sha, train2_X_M_exp, train2_X_R_sha, train2_X_R_exp],
            input_X3=[train3_X_M_sha, train3_X_M_exp, train3_X_R_sha, train3_X_R_exp],
            n_modes=6, 
            Pheno=[Pheno_t0_Multi_sha, Pheno_t0_Multi_exp, Pheno_t0_Rosea_sha, Pheno_t0_Rosea_exp],
            Pheno_t=[Pheno_t1_Multi_sha, Pheno_t1_Multi_exp, Pheno_t1_Rosea_sha, Pheno_t1_Rosea_exp],
            outputs=[np.concatenate((train1_Y_M_sha, train2_Y_M_sha, train3_Y_M_sha), axis=1),
                    np.concatenate((train1_Y_M_exp, train2_Y_M_exp, train3_Y_M_exp), axis=1),
                    np.concatenate((train1_Y_R_sha, train2_Y_R_sha, train3_Y_R_sha), axis=1),
                    np.concatenate((train1_Y_R_exp, train2_Y_R_exp, train3_Y_R_exp), axis=1)],
            input_Y1=[train1_Y_M_sha, train1_Y_M_exp, train1_Y_R_sha, train1_Y_R_exp],
            input_Y2=[train2_Y_M_sha, train2_Y_M_exp, train2_Y_R_sha, train2_Y_R_exp],
            input_Y3=[train3_Y_M_sha, train3_Y_M_exp, train3_Y_R_sha, train3_Y_R_exp],
            input_df1=[MFsha_trans3, MFexp_trans3, Roseasha_trans3, Roseaexp_trans3],
            input_df2=[MFsha_prot3, MFexp_prot3, Roseasha_prot3, Roseaexp_prot3],
            input_df3=[MFsha_met, MFexp_met, Roseasha_met, Roseaexp_met],
            comparison=["M_s_vs_c", "R_s_vs_c", "R_vs_M_s", "R_vs_M_c"], iter=int(h), 
            products=products, names=names2, features=features, features2=names
        )

        return grid, control_sensitivity, G1, G2, names, cd, red_feat_space1, red_feat_space2


def replicate_noise(max_iter, task_id):
    
    np.random.seed(os.getpid())
    np.random.seed(None)
    random.seed(None)
    
    results = Parallel(n_jobs=32, backend="loky")(delayed(run_iteration)(h, feature_labels) for h in range(max_iter))

    grids = []
    sens = []
    G1_list = [] 
    G2_list = []
    r2_outs = []
    red_space1 = []
    red_space2 = []
    mode1 = []
    mode2 = []
    
    for result in results:
        grid, sensitivity_matrix, G1, G2, names, cd, red_feat_space1, red_feat_space2 = result
        grids.append(grid)
        sens.append(sensitivity_matrix)
        G1_list.append(G1)
        G2_list.append(G2)
        r2_outs.append(cd)
        red_space1.append(red_feat_space1)
        red_space2.append(red_feat_space2)
 
    all_results = []
    
    for result_df in grids:
        for column in result_df.columns:
            result_df[column] = result_df[column].apply(convert_string_to_list)

        all_results.append(result_df)
    
    grids = pd.concat(all_results, ignore_index = True)
    grids = grids.applymap(lambda x: x[0] if isinstance(x, list) else x)
    index = [p for p in grids.index][:int(len(grids.index)/2)] * 2
    # result = [f"{x}{y}" for x, y in zip(grids['kernel'], index)]
    # grids['kernel'] = result
        
    result_df = grids.groupby(['kernel', 'gamma', 'coef', 'degree', 'gamma1', 'delta1', 'coef2', 'coef3', 'l', 'retained']).agg({
        'r2_data multi (control)': ['mean', 'std'], 
        'r2_data multi (stress)': ['mean', 'std'],
        'r2_data rosea (control)': ['mean', 'std'],
        'r2_data rosea (stress)': ['mean', 'std']
            }).reset_index()

    result_df.to_csv("iteration_stats_all.csv") ## data reconstruction accuracies for each kernel 

    sens_cleaned = [df.reset_index(drop=False) for df in sens]

    sens = pd.concat(sens_cleaned, keys=[f'DF_{i}' for i in range(len(sens))], ignore_index=False)

    df_mean = sens.groupby(sens.columns[0]).mean()
    df_mean = df_mean.sort_values(df_mean.columns[0], ascending=False)
    df_mean = df_mean[df_mean.index != "NA"]
    names_1 = df_mean.index.to_series().str.replace(r'\([^)]*?_','(', regex=True)
    df_mean = np.array(df_mean.iloc[:, 1]).reshape(-1, 1)
    
    fig_width_in = 10     # adjust width in inches (between 6.68 and 19.05)
    fig_height_in = 25    # adjust height in inches but can scale dynamically if needed
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    fig.patch.set_facecolor('white')

    sns.heatmap(df_mean, annot=False, cmap='coolwarm', cbar=True, yticklabels=names_1, ax=ax)

    plt.xticks([])
    plt.tick_params(axis='y', labelsize=14)  

    ax.set_aspect(0.1)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Sensitivity values', fontsize=14)
    # Title
  #  plt.title('Mean Sensitivity Heatmap', fontsize=12, fontproperties=times_font)

    plt.tight_layout(pad=3.0)

    plt.savefig('Figure 5C.eps', format='eps', dpi=600)
    plt.show()
    plt.close(fig)
    
    # grouped eigenmode topologies
    G1_list = [G1 for G1_sub in G1_list for G1 in G1_sub]
    G2_list = [G2 for G2_sub in G2_list for G2 in G2_sub]
    
    if len(G1_list) > 1 and len(G2_list) > 1:

        F1_freq, W1_sum = aggregate_edge_frequency(G1_list)
        
        G1_consensus = threshold_by_frequency(F1_freq, thr=0.5)
            
        nodes = names # G1_consensus.nodes()
        
        A1 = nx.to_numpy_array(G1_consensus, nodelist=sorted(G1_consensus.nodes()), weight='weight', dtype=float)
        A1 = pd.DataFrame(A1, index=names, columns=names)
        edges = A1.reset_index().melt(id_vars='index', var_name='target', value_name='weight')
        edges = edges.rename(columns={'index': 'source'})
        
        edges = edges[edges['weight'] != 0]
        edges = edges[edges['source'] != edges['target']]
        edges['abs_weight'] = np.abs(edges['weight'])
        edges.sort_values(by="weight", ascending=False).iloc[:100]
        
        pathway_dict = {re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', str(val)).strip(): feature_labels['Pathway'].iloc[idx]
                            for idx, val in enumerate(feature_labels['Name'])}
            
        product_dict = {re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', str(val)).strip(): feature_labels['Product'].iloc[idx]
                            for idx, val in enumerate(feature_labels['Name'])}
            
        edges['Source_names'] = edges['source']
        edges['Target_names'] = edges['target']
        edges.to_csv(f'S1 Table.csv', index=False)

        F2_freq, W1_sum = aggregate_edge_frequency(G2_list)
        G2_consensus = threshold_by_frequency(F2_freq, thr=0.5)
            
        #nodes = G2_consensus.nodes()
        A2 = nx.to_numpy_array(G2_consensus, nodelist=sorted(G2_consensus.nodes()), weight='weight', dtype=float)
        A2 = pd.DataFrame(A2, index=names, columns=names)
        edges = A2.reset_index().melt(id_vars='index', var_name='target', value_name='weight')
        edges = edges.rename(columns={'index': 'source'})
        
        edges = edges[edges['weight'] != 0]
        edges = edges[edges['source'] != edges['target']]
        edges['abs_weight'] = np.abs(edges['weight'])
        edges.sort_values(by="weight", ascending=False).iloc[:100]
        edges['Source_names'] = edges['source']
        edges['Target_names'] = edges['target']
        edges.to_csv(f'S2 Table.csv', index=False)
    
    if len(r2_outs) > 1:
        r2_outs = np.asarray(r2_outs)
        r2_mean = np.mean(r2_outs, axis=0)
        r2_std = np.std(r2_outs, axis=0)
        
        with open('r2_outs.txt', 'w') as f: ## mean for reconstruction accuracy of phenocopying Rosea outputs 
            f.write(f"r2_outs_mean: {r2_mean}")
            f.write(f"r2_outs_std: {r2_std}")
            f.write(f"r2_outs_shape: {r2_outs.shape}")
    
    
    combined_df1 = pd.concat(red_space1, ignore_index=True) if red_space1 else pd.DataFrame()
    combined_df2 = pd.concat(red_space2, ignore_index=True) if red_space2 else pd.DataFrame()
    cmap = cm.Blues
    fig, axs = plt.subplots(1, 2, figsize=(25, 9))  
    norm = plt.Normalize(vmin=0, vmax=1) 
    plot_species_barplot(combined_df1, axs[0], r'$\it{C.\ major}$', cmap, norm)  
    plot_species_barplot(combined_df2, axs[1], r'$\it{C.\ rosea}$', cmap, norm)  
    plt.subplots_adjust(right=1)
    plt.tight_layout()
    plt.savefig("Figure 5A.eps", dpi=400, bbox_inches="tight", facecolor="white")
    plt.show()
                
    return 


reported_grid = replicate_noise(task_id='kernel_replication_1', max_iter=20)
