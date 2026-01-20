import numpy as np
from sklearn.metrics import r2_score
import os 
from scipy.stats import pearsonr

np.random.seed(8)  
os.environ["OMP_NUM_THREADS"] = "1" 

def modal_reconstruction(C, outs, eVals_r, modes, b_r0, ntimepts=None, nmodes=None, n_phenos=None, mode_pairs=None):
    """
    Output reconstruction using each retained kernel DMD eigenmode 

    Parameters:
    - C (numpy array): output matrix C that maps inputs to outputs identified from multi-objective 
    - outs (numpy array): phenotype outputs
    - eVals_r (numpy array): eigenvalues 
    - modes (numpy array): eigenmodes
    - b_r0 (numpy array): eigenmode amplitudes
    - ntimepts: number of timepoints 
    - nmodes: number of eigenmodes
    - n_phenos: number of phenotypes
    - mode_pairs: eigenmode conjugate pairs
        
    Returns:
    - outputs2: appended list of output reconstructions for each eigenmode
    - cd_modes: correlation coefficient between predicted and original output distributions for each eigenmode 
    - r2_modes: coefficient of determination (R2) between predicted and original output distributions for each eigenmode 
    - relative_error: standardized l1 norm difference between predicted and original output distributions for each eigenmode 
    """
    cd_modes = []
    outputs2 = []
    r2_modes = []
    modal_error = []

    if nmodes < len(mode_pairs):
        length = nmodes
    else:
        length = len(mode_pairs)

    for g in range(length):
        scaled_mode = eVals_r[mode_pairs[g][0]] * np.asarray(b_r0)[mode_pairs[g][0], :]
        cd = np.zeros((outs.shape[0]))
        relative_error = np.zeros((outs.shape[0]))
        feature_means = np.zeros((outs.shape[0]))
        r2 = np.zeros((outs.shape[0]))
        outputs1 = np.zeros((n_phenos, ntimepts))
        outputs1[:, 0] = np.asarray(outs[:, 0])
        for i in range(1, ntimepts):
            outputs1[:, i] = (C @ np.real(modes[:, mode_pairs[g][0]] * scaled_mode[i - 1])).reshape(n_phenos, )
        for k in range(outs.shape[0]):
            feature_means[k] = np.mean(np.asarray(outs[k, :ntimepts]))
            cd[k] = np.corrcoef(np.asarray(outs[k, :ntimepts]), outputs1[k, :ntimepts])[0, 1] ** 2
            r2[k] = r2_score(outs[k, :ntimepts], outputs1[k, :ntimepts])
            relative_error[k] = np.linalg.norm(outs[k, :ntimepts] - outputs1[k, :ntimepts])**2 / np.linalg.norm(outs[k, :ntimepts])**2
        cd_modes.append(cd)
        r2_modes.append(r2)
        modal_error.append(relative_error)
        outputs2.append(outputs1)
        
    return outputs2, cd_modes, r2_modes, modal_error

def reconstructed_data(K, inputs1, ntimepts=None):
    """
    Linear Data reconstruction using the linear operator from kernel DMD 

    Parameters:
    - K (numpy array): linear (Koopman) operator 
    - inputs (numpy array): concatenated inputs 
    - ntimepts: number of timepoints 
        
    Returns:
    - snapshots: reconstructed inputs
    - cd: correlation coefficient between predicted and original input distributions
    - mse: mean squared error between predicted and original input distributions
    - r2: coefficient of determination (R2) between predicted and original input distributions
    """
    snapshots = np.zeros((inputs1.shape[1], ntimepts))
    snapshots[:, 0] = inputs1.T[:, 0]
    
    for i in range(1, ntimepts):
        previous_snapshot = np.asarray(inputs1[i - 1, :].reshape(1, -1))
        snapshots[:, i] = (K).dot(previous_snapshot.T).reshape(inputs1.shape[1], ) 
                
    cd = np.corrcoef(np.asarray(inputs1.T[:, :ntimepts]), snapshots)[0, 1] ** 2
    r2 = r2_score(inputs1.T[:, :ntimepts], snapshots)
    mse = 1 / ntimepts * np.linalg.norm(snapshots - np.asarray(inputs1.T[:, :ntimepts]), ord=2) ** 2 / np.linalg.norm(
        np.asarray(inputs1.T[:, :ntimepts]), ord=2)

    return snapshots, cd, mse, r2

def reconstructed_outputs(K, C, outs, inputs, ntimepts=None, n_phenos=None):
    """
    Linear output reconstruction using the linear operators from kernel DMD 

    Parameters:
    - K (numpy array): linear (Koopman) operator that maps inputs forward in time 
    - C (numpy array): output matrix that maps inputs to outputs 
    - inputs (numpy array): concatenated inputs 
    - ntimepts: number of timepoints 
    - n_phenos: number of phenotypes
        
    Returns:
    - snapshots: reconstructed inputs
    - cd: correlation coefficient between predicted and original output distributions
    - mse: mean squared error between predicted and original output distributions
    - r2: coefficient of determination (R2) between predicted and original output distributions
    - relative_error: standardized l1 norm difference between predicted and original output distributions
    """
    outputs1 = np.zeros((n_phenos, ntimepts))
    outputs1[:, 0] = np.asarray(outs[:, 0])
    cd = np.zeros((n_phenos))
    r2 = np.zeros((n_phenos))
    relative_error =  np.zeros((n_phenos))
    feature_means = np.zeros((n_phenos))
    
    for i in range(1, ntimepts):
        previous_snapshot = inputs[i - 1, :].reshape(1, -1)
        outputs1[:, i] = (C @ (np.asarray(K).dot(previous_snapshot.T))).reshape(n_phenos, )

    for j in range(outs.shape[0]):
        feature_means[j] = np.mean(np.asarray(outs[j, :ntimepts]))
        cd[j] = np.corrcoef(np.asarray(outs[j, :ntimepts]), outputs1[j, :ntimepts])[0, 1] ** 2
        r2[j] = r2_score(outs[j, :ntimepts], outputs1[j, :ntimepts])
        relative_error[j] = np.linalg.norm(outs[j, :ntimepts] - outputs1[j, :ntimepts])**2 / np.linalg.norm(outs[j, :ntimepts])**2

    mse = 1 / ntimepts * np.linalg.norm(outputs1 - np.asarray(outs[:, :ntimepts]), ord=2) ** 2 / np.linalg.norm(
            np.asarray(outs[:, :ntimepts]), ord=2)
    
    return outputs1, cd, mse, r2, relative_error

def reconstructed_outputs_MPC(K, C, outs, inputs, B, u, ntimepts=None, n_phenos=None):
    """
    Linear output reconstruction with control for evaluating how well the optimized inputs of one species predict the outputs of another species after MPC

    Parameters:
    - K (numpy array): linear (Koopman) operator that maps inputs forward in time 
    - C (numpy array): output matrix that maps inputs to outputs 
    - outs (numpy array): concatenated outputs
    - inputs (numpy array): optimized inputs after MPC 
    - B (numpy array): control input matrix 
    - u (numpy array): optimized control inputs after MPC
    - ntimepts: number of timepoints 
    - n_phenos: number of phenotypes
        
    Returns:
    - outputs1: reconstructed outputs
    - cd: correlation coefficient between predicted and original output distributions
    - mse: mean squared error between predicted and original output distributions
    - r2: coefficient of determination (R2) between predicted and original output distributions
    """
    # C => output mapping function, W => kernel weights, inputs => concatenated inputs
    outputs1 = np.zeros((n_phenos, ntimepts))
    outputs1[:, 0] = np.asarray(outs[:, 0])
    cd = np.zeros((n_phenos))
    r2 = np.zeros((n_phenos))
    feature_means = np.zeros((n_phenos))
    for i in range(1, ntimepts):
        previous_snapshot = inputs[:, i - 1].reshape(-1, 1)
        u_previous = u[:, i - 1].reshape(-1, 1)
        outputs1[:, i] = (C @ ((K @ previous_snapshot) + (B @ u_previous))).reshape(n_phenos, )
    for j in range(n_phenos):
        feature_means[j] = np.mean(outs[j, :ntimepts])
        corr, _ = pearsonr(outs[j, :ntimepts], outputs1[j, :ntimepts]) # [0, 1] ** 2 # pearsonr
        cd[j] = corr
        r2[j] = r2_score(outs[j, :ntimepts], outputs1[j, :ntimepts])
        mse = 1 / ntimepts * np.linalg.norm(outputs1 - outs[:, :ntimepts], ord=2) ** 2 / np.linalg.norm(
                outs[:, :ntimepts], ord=2)
    return outputs1, cd, mse, r2