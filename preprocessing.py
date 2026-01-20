import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')  
from sklearn.preprocessing import StandardScaler
from itertools import chain
from itertools import combinations
import os 
import networkx as nx
import ast

scaler2 = StandardScaler()

np.random.seed(None)  
#os.environ["OMP_NUM_THREADS"] = "1" 

def timepoint_rearrangements(data1, data2, data3, output1, output2, output3, nreps=6, ntps=4):
    """
    Add random noise to each sample and rearrange replicates in a timeseries

    Parameters:
    - data1, data2, data3 (pandas DataFrame): input data of different modality (transcriptome, proteome, metabolome) where rows are samples and columns are features.
    - output1, output2, output3 (pandas DataFrame): output data 
    - nreps: number of replicates
    - ntps: number of timepoints
        
    Returns:
    - standardized input and output data in the right order of samples
    """
    # inputs
    data_red1 = data1.iloc[:12, :]
    data_red2 = data2.iloc[:12, :]
    data_red3 = data3.iloc[:12, :]
    data_red1_1 = []
    data_red2_1 = []
    data_red3_1 = []
    output1_1 = []
    output2_1 = []
    for i in range(data_red1.shape[0]):
        data_red1_1.append(data_red1.iloc[i, :])
        data_red1_1.append(data_red1.iloc[i, :] + int(np.abs(np.random.normal(0, 10))))
        data_red2_1.append(data_red2.iloc[i, :])
        data_red2_1.append(data_red2.iloc[i, :] + np.abs(np.random.normal(0, 0.0001)))
        data_red3_1.append(data_red3.iloc[i, :])
        data_red3_1.append(data_red3.iloc[i, :] + np.abs(np.random.normal(0, 0.01)))
        output1_1.append(output1.iloc[i, :])
        output1_1.append(output1.iloc[i, :] + np.abs(np.random.normal(0, 0.01)))
        output2_1.append(output2.iloc[i, :])
        output2_1.append(output2.iloc[i, :] + np.abs(np.random.normal(0, 0.0001)))
    data_red1_1 = pd.DataFrame(scaler2.fit_transform(np.asarray(data_red1_1)))
    data_red2_1 = pd.DataFrame(scaler2.fit_transform(np.asarray(data_red2_1)))
    data_red3_1 = pd.DataFrame(scaler2.fit_transform(np.asarray(data_red3_1)))
    output1_1 = pd.DataFrame(scaler2.fit_transform(output1_1))
    output2_1 = pd.DataFrame(output2_1).apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)
    # timepoint rearrangements
    tps = []
    for k in range(0, nreps * ntps, nreps):
        tps.append([[i for i in range(nreps)][i] + k for i in range(nreps)])
    index = pd.DataFrame(tps).melt()["value"].values.tolist()
    index_0 = index[:-1]
    index_1 = index[1:]
    train1_X = np.zeros(shape=(len(index_0), data_red1_1.shape[1]))
    train2_X = np.zeros(shape=(len(index_0), data_red2_1.shape[1]))
    train3_X = np.zeros(shape=(len(index_0), data_red3_1.shape[1]))
    train1_Y = np.zeros(shape=(len(index_1), data_red1_1.shape[1]))
    train2_Y = np.zeros(shape=(len(index_1), data_red2_1.shape[1]))
    train3_Y = np.zeros(shape=(len(index_1), data_red3_1.shape[1]))
    for ii in range(train1_X.shape[0]):
        train1_X[ii] = data_red1_1.iloc[index_0[ii], :]
        train2_X[ii] = data_red2_1.iloc[index_0[ii], :]
        train3_X[ii] = data_red3_1.iloc[index_0[ii], :]
        train1_Y[ii] = data_red1_1.iloc[index_1[ii], :]
        train2_Y[ii] = data_red2_1.iloc[index_1[ii], :]
        train3_Y[ii] = data_red3_1.iloc[index_1[ii], :]
    # outputs
    Out1 = pd.DataFrame(np.asarray(output1_1).reshape(24, 1))
    output_0 = np.zeros(shape=(len(index_0), Out1.shape[1]))
    output_1 = np.zeros(shape=(len(index_1), Out1.shape[1]))
    for ii in range(output_0.shape[0]):
        output_0[ii] = Out1.iloc[index_0[ii], :]
    for ii in range(output_1.shape[0]):
        output_1[ii] = Out1.iloc[index_1[ii], :]
    Malate_t0 = np.row_stack(output_0)
    Malate_t1 = np.row_stack(output_1)
    CAM_pheno = pd.DataFrame(np.asarray(output2_1).reshape(24, 4))
    output_0 = np.zeros(shape=(len(index_0), CAM_pheno.shape[1]))
    output_1 = np.zeros(shape=(len(index_1), CAM_pheno.shape[1]))
    for ii in range(output_0.shape[0]):
        output_0[ii] = CAM_pheno.iloc[index_0[ii], :]
    for ii in range(output_1.shape[0]):
        output_1[ii] = CAM_pheno.iloc[index_1[ii], :]
    CAM_t0 = np.row_stack(output_0)
    CAM_t1 = np.row_stack(output_1)
    Pheno_gas0 = output3.iloc[:len(index_0), :].apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)
    Pheno_gas1 = output3.iloc[1:len(index_1) + 1, :].apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)
    Pheno_t0 = np.column_stack((Malate_t0, np.asarray(Pheno_gas0), CAM_t0))
    Pheno_t1 = np.column_stack((Malate_t1, np.asarray(Pheno_gas1), CAM_t1))
    
    return np.row_stack(train1_X), np.row_stack(train2_X), np.row_stack(train3_X), np.row_stack(train1_Y), np.row_stack(
        train2_Y), np.row_stack(train3_Y), Pheno_t1, Pheno_t0

def add_missing_columns(df, cols):
    """
    0-pad missing columns to a dataframe

    Parameters:
    - df (pandas DataFrame): dataframe to which 0-columns will be added 
    - cols (list): list of column indices that are missing from df
        
    Returns:
    - the dataframe and added missing columns with 0 values 
    """
    for col in cols:
        df[col] = 0
    return df

def scale_and_transform(data, scaler):
    """
    Scale data

    Parameters:
    - data (numpy array)
    - scaler: scaler of choice (StandardScaler)
        
    Returns:
    - scaled data 
    """
    return pd.DataFrame(scaler.fit_transform(data.sort_index(axis=1)), columns=data.columns)


def get_important_features(data, n_components=6, top_n=60):
    """
    PCA of concatenated input data and feature selection of features with high loading scores

    Parameters:
    - data (numpy array): concatenated multiome data array 
    - n_components: number of PCA components to keep 
    - top_n: top number of features to retain in terms of loading score
        
    Returns:
    - list of indices of top_n features 
    """
    pca = PCA(svd_solver='arpack', random_state=3)
    pca.fit(data)
    
    return list(chain.from_iterable(
        [pd.DataFrame(np.abs(pca.components_[i]).argsort()[::-1])[0].values.tolist()[:top_n // (i + 1)] for i in range(n_components)]
    ))
    
def get_cam_features(transcriptome, proteome, cam_values):
    """
    select just the CAM-related features from input data 

    Parameters:
    - transcriptome (numpy array)
    - proteome (numpy array)  
    - cam_values: CAM feature labels
        
    Returns:
        input data with just CAM-related features  
    """
    tran_features = [x for x in transcriptome.columns if re.sub("values.", "", x) in cam_values]
    prot_features = [x for x in proteome.columns if re.sub("values.", "", x) in cam_values]
    tran_index = np.where(transcriptome.columns.isin(tran_features))[0].tolist()
    prot_index = np.where(proteome.columns.isin(prot_features))[0].tolist()
    
    return transcriptome.iloc[:, tran_index], proteome.iloc[:, prot_index]


def sort_and_concat(input1, input2):
    """
    sort the column indices of input data, concatenate and remove duplicate entries 

    Parameters:
    - input1, input2 (numpy arrays)

    Returns:
    - dataframe with all relevant features for analysis  
    """
    df = pd.concat((input1.sort_index(axis=1), input2.sort_index(axis=1)), axis=1)
    
    return df.loc[:, ~df.columns.duplicated()]

def make_int_matrix(names):
    """
    Creates an adjacency matrix from interactions between features based on protein-protein networks (STRING) and protein-metabolite networks (KEGG)

    Parameters:
    - names: feature names

    Returns:
    - adjacency matrix
    """
    interactions = pd.read_csv('filtered_interactions.csv')

    interaction_dict = {
        tuple(sorted([row.Entity_A, row.Entity_B])): row.Score
        for row in interactions.itertuples(index=False)
        if row.Score > 0.9
    }
    
    retained_interactions = []

    for i, i2 in combinations(names, 2):
        pair = tuple(sorted([i, i2]))
        score = 1.0 if pair in interaction_dict else 0.0
        retained_interactions.append({'Source': i, 'Target': i2, 'value': score})
        retained_interactions.append({'Source': i2, 'Target': i, 'value': score})  # enforce symmetry

    retained_interactions_df = pd.DataFrame(retained_interactions)

    retained_cleaned = retained_interactions_df.groupby(['Source', 'Target'], as_index=False).agg({'value': 'max'})
    adjacency_matrix = retained_cleaned.pivot(index='Source', columns='Target', values='value')
    adjacency_matrix = adjacency_matrix.reindex(index=names, columns=names, fill_value=0.0)

    return adjacency_matrix.values

def aggregate_edge_frequency(adj_matrices):
    """
    adj_matrices: list of N x N numpy arrays (binary or weighted). 
                  Nonzero means edge present.
    Returns:
      F_freq: N x N array of frequencies (0..1)
      W_sum:  N x N array of summed weights
    """
    T = len(adj_matrices)
    N = adj_matrices[0].shape[0]
    F = np.zeros((N, N), dtype=float)
    W = np.zeros((N, N), dtype=float)
    for A in adj_matrices:
        present = (np.abs(A) > np.quantile(A, 0.8))
        F += present.astype(float)
        W += A  # if A is binary, this sums 1s; else sums weights
    F_freq = F / float(T)
    return F_freq, W

def threshold_by_frequency(F_freq, thr=0.5):
    """
    Keep edges with frequency >= thr. Returns NetworkX graph.
    """
    N = F_freq.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    u,v = np.where(F_freq >= thr)
    for i,j in zip(u,v):
        if i < j:  # undirected
            G.add_edge(i, j, freq=F_freq.iloc[i,j])
    return G

def convert_string_to_list(value):
    try:
        # Try to evaluate the string as a Python literal (e.g., a list, tuple, etc.)
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Return the value as is if it's not a valid list-like string
        return value

