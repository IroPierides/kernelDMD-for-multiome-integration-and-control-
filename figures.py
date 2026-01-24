import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from scipy.optimize import linear_sum_assignment
import os 
from matplotlib import rcParams
from matplotlib import font_manager as fm
times_font = fm.FontProperties(fname="times.ttf")
from matplotlib import cm
import pandas as pd

#np.random.seed(8)  
#os.environ["OMP_NUM_THREADS"] = "1" 

def match_modes_by_eigenvalue_distance(eVals_1, mode_pairs1, b_1, eVals_4, mode_pairs4, b_4):
    """
    Match mode_pairs1 to mode_pairs4 one-to-one using minimal eigenvalue modulus distance.
    Uses Hungarian algorithm to find optimal assignment.
    """
    # Use only the primary eigenvalue from each pair (first index in each pair)
    target_eigs = np.array([eVals_1[pair[0]] for pair in mode_pairs1])
    ref_eigs = np.array([eVals_4[pair[0]] for pair in mode_pairs4])

    # Build cost matrix: abs(modulus difference)
    cost_matrix = np.abs(ref_eigs[:, np.newaxis] - target_eigs[np.newaxis, :])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder mode_pairs1 and eVals_1 based on assignment
    unassigned_indices1 = [i for i in range(len(mode_pairs1)) if i not in col_ind]
    # Extend the reordered_pairs2 list with  unassigned mode_pairs
    reordered_pairs1 = [mode_pairs1[j] for j in col_ind]
    reordered_pairs1.extend([mode_pairs1[i] for i in unassigned_indices1])
    reordered_eVals_1 = []
    reordered_b_1 = []
    
    unassigned_indices = [i for i in range(len(mode_pairs4)) if i not in row_ind]
    reordered_pairs2 = [mode_pairs4[j] for j in row_ind]
    reordered_pairs2.extend([mode_pairs4[i] for i in unassigned_indices])
    reordered_eVals_2 = []
    reordered_b_2 = []
    
    for pair in reordered_pairs1:
        if len(pair) == 2:
            reordered_eVals_1.append([eVals_1[pair[0]], eVals_1[pair[1]]])
            reordered_b_1.append(b_1[pair[0]])
        elif len(pair) == 1:
            reordered_eVals_1.append([eVals_1[pair[0]]])
            reordered_b_1.append(b_1[pair[0]])
    
    for pair in reordered_pairs2:
        if len(pair) == 2:
            reordered_eVals_2.append([eVals_4[pair[0]], eVals_4[pair[1]]])
            reordered_b_2.append(b_4[pair[0]])
        elif len(pair) == 1:
            reordered_eVals_2.append([eVals_4[pair[0]]])
            reordered_b_2.append(b_4[pair[0]])
            
    return reordered_eVals_1, reordered_pairs1, reordered_b_1, reordered_eVals_2, reordered_pairs2, reordered_b_2, col_ind, row_ind


def mode_clusters_fig(eVals_1, eVals_4, mode_pairs1, mode_pairs4, b_1, b_4, label1,  label4, figlabel):
    """
    Generate and save eigendecomposition plots

    Parameters:
        - eVals_1, eVals_4: eigenvalues
        - mode_pairs1, mode_pairs4: eigenmode pairs
        - b_1, b_4: eigenmode amplitudes
        - label1,  label4: species labels
        - figlabel: figure label
    
    Returns:
        - col_ind1, col_ind2: reordered eigenvalue and eigenmode indices
        
    """
    eVals_1, mode_pairs1, b_1, eVals_4, mode_pairs4, b_4, col_ind1, col_ind2 = match_modes_by_eigenvalue_distance(eVals_1, mode_pairs1, b_1, eVals_4, mode_pairs4, b_4)

    label1 = r'$\it{C.\ major}$'
    label4 = r'$\it{C.\ rosea}$'
    my_figsize = (15, 8)

    cmap = matplotlib.cm.get_cmap('Greys')
    matplotlib.rcParams['axes.grid'] = False

    corr_tspan = range(0, 12, 1)
    k = 0
    k1 = 0

    length = max(len(mode_pairs1), len(mode_pairs4))

    fig1, axs1 = plt.subplots(2, 4, figsize=my_figsize, sharex=False)

    markers = [
    'o',  's',  'd',  '^',  'v',  '>', '<', 'p',  'P',  '*',  'h',  'H',  'X',  'x',  '+',  'D',  '|',  '_',  '1',   '2',  '3',  '4',  ]
    theta = np.linspace(0, 2 * np.pi, 200)
    axs1[0][0].plot(np.cos(theta), np.sin(theta), c='black', lw=1.5)
    for ii in range(len(mode_pairs1)):
        axs1[0][0].plot(np.real(eVals_1[ii][0]), np.imag(eVals_1[ii][0]),
                        marker=markers[ii], mfc="#008080", mec="black", mew=1.2, ms=8) ##008080
        if len(mode_pairs1[ii]) == 2:
            axs1[0][0].plot(np.real(eVals_1[ii][1]), np.imag(eVals_1[ii][1]),
                            marker=markers[ii], mfc="#008080", mec="black", mew=1.2, ms=8)

    axs1[1][0].plot(np.cos(theta), np.sin(theta), c='black', lw=1.5)
    for ii in range(len(mode_pairs4)):
        axs1[1][0].plot(np.real(eVals_4[ii][0]), np.imag(eVals_4[ii][0]),
                        marker=markers[ii], mfc="Gold", mec="black", mew=1.2, ms=8)
        if len(mode_pairs4[ii]) == 2:
            axs1[1][0].plot(np.real(eVals_4[ii][1]), np.imag(eVals_4[ii][1]),
                            marker=markers[ii], mfc="Gold", mec="black", mew=1.2, ms=8)

    axs1[0][0].set_xlabel(r'Re$(\lambda)$', fontsize=10, fontproperties=times_font)
    axs1[0][0].set_ylabel(r'Im$(\lambda)$', fontsize=10, fontproperties=times_font)
    axs1[0][0].set_xlim(-1.2, 1.2)
    axs1[0][0].set_xticks(np.arange(-1.2, 1.2, 0.5))
    axs1[0][0].set_ylim(-1.2, 1.2)
    axs1[0][0].set_yticks(np.arange(-1.2, 1.2, 0.5))
    axs1[0][0].xaxis.set_tick_params(which='both', size=0.1, width=1.5, direction='out', labelsize=8)
    axs1[0][0].yaxis.set_tick_params(which='both', size=0.1, width=1.5, direction='out', labelsize=8)

    axs1[1][0].set_xlabel(r'Re$(\lambda)$', fontsize=10, fontproperties=times_font)
    axs1[1][0].set_ylabel(r'Im$(\lambda)$', fontsize=10, fontproperties=times_font)
    axs1[1][0].set_xlim(-1.2, 1.2)
    axs1[1][0].set_xticks(np.arange(-1.2, 1.2, 0.5))
    axs1[1][0].set_ylim(-1.2, 1.2)
    axs1[1][0].set_yticks(np.arange(-1.2, 1.2, 0.5))
    axs1[1][0].xaxis.set_tick_params(which='both', size=0.1, width=1.5, direction='out', labelsize=8)
    axs1[1][0].yaxis.set_tick_params(which='both', size=0.1, width=1.5, direction='out', labelsize=8)

    for label in axs1[0][0].get_xticklabels() + axs1[0][0].get_yticklabels() + axs1[1][0].get_xticklabels() + axs1[1][0].get_yticklabels():
        label.set_fontproperties(times_font)
    
    markers2 = [
    'o-', 's-', 'd-', '^-', 'v-', '>-', '<-',  'p-', 'P-',  '*-',  'h-',  'H-',  'X-',  'x-', '+-',  'D-',  '|-', '_-', '1-', '2-', '3-',  '4-',  ]
    p = 1
    for i1 in range(1, len(mode_pairs1) + 1, 1):
        k = i1 - 1
        scaled_mode1 = []
        for i2 in range(12):
            ampl = np.asarray(b_1[k][0])
            if np.allclose(ampl.imag, 0, atol=1e-10):
                ampl = ampl.real
            scaled_mode1.append(np.real(np.power(eVals_1[k][0], i2) * ampl))
        axs1[0][0].set_title(label1, fontsize=12, loc='left', fontproperties=times_font)
        if np.abs(eVals_1[k][0]) < 0.95 and np.imag(eVals_1[k][0]) < 0.1: 
            axs1[0][1].plot(corr_tspan, np.asarray(scaled_mode1), markers2[k], ms=8, mec="#008080", mew=1, c="#008080",
                        mfc="#008080", lw=1.2 )
            axs1[0][1].set_ylabel('Stable, decaying Mode dynamics', fontsize=10, fontproperties=times_font)
        elif np.abs(eVals_1[k][0]) > 0.95 and np.imag(eVals_1[k][0]) == 0: 
            axs1[0][2].plot(corr_tspan, np.asarray(scaled_mode1), markers2[k], ms=8, mec="#008080", mew=1, c="#008080",
                        mfc="#008080", lw=1.2 )
            axs1[0][2].set_ylabel('Sustainable Mode dynamics', fontsize=10, fontproperties=times_font)
        elif np.imag(eVals_1[k][0]) > 0.1: 
            axs1[0][3].plot(corr_tspan, np.asarray(scaled_mode1), markers2[k], ms=8, mec="#008080", mew=1, c="#008080",
                        mfc="#008080", lw=1.2 )
            axs1[0][3].set_ylabel('Oscillatory Mode dynamics', fontsize=10, fontproperties=times_font)
        axs1[0][1].set_ylim(-40, 40)
        axs1[0][1].set_xticks(range(0, 12, 1))
        axs1[0][1].set_xticklabels(['4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00'], rotation=45)
        axs1[0][2].set_ylim(-40, 40)
        axs1[0][2].set_xticks(range(0, 12, 1))
        axs1[0][2].set_xticklabels(['4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00'], rotation=45)
        axs1[0][3].set_ylim(-40, 40)
        axs1[0][3].set_xticks(range(0, 12, 1))
        axs1[0][3].set_xticklabels(['4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00'], rotation=45)
        for jj, ax in enumerate(axs1.reshape(-1)):
            ax.xaxis.set_tick_params(which='both', size=3, width=1.5, direction='out', labelsize=8)
            ax.yaxis.set_tick_params(which='both', size=3, width=1.5, direction='out', labelsize=8)
        plt.tight_layout() 
        k = k + 1
            
    q2 = 1
    for i3 in range(1, len(mode_pairs4) + 1, 1):
        k1 = i3 - 1
        scaled_mode4 = []
        for i4 in range(12):
            ampl2 = np.asarray(b_4[k1][0])
            if np.allclose(ampl2.imag, 0, atol=1e-10):
                ampl2 = ampl2.real
            scaled_mode4.append(np.real(np.power(eVals_4[k1][0], i4) * ampl2))
        axs1[1][0].set_title(label4, fontsize=12, loc='left', fontproperties=times_font)
        if np.abs(eVals_4[k1][0]) < 0.9 and np.imag(eVals_4[k1][0]) < 0.1: 
            axs1[1][1].plot(corr_tspan, np.asarray(scaled_mode4), markers2[k1], ms=8, mec="Gold", mew=1, c="Gold",
                        mfc="Gold", lw=1.2 )
            axs1[1][1].set_ylabel('Stable, decaying Mode dynamics', fontsize=10, fontproperties=times_font)
        elif np.abs(eVals_4[k1][0]) > 0.9 and np.imag(eVals_4[k1][0]) == 0: 
            axs1[1][2].plot(corr_tspan, np.asarray(scaled_mode4), markers2[k1], ms=8, mec="Gold", mew=1, c="Gold",
                        mfc="Gold", lw=1.2 )
            axs1[1][2].set_ylabel('Sustainable Mode dynamics', fontsize=10, fontproperties=times_font)
        elif np.imag(eVals_4[k1][0]) > 0.1: 
            axs1[1][3].plot(corr_tspan, np.asarray(scaled_mode4), markers2[k], ms=8, mec="Gold", mew=1, c="Gold",
                        mfc="Gold", lw=1.2 )
            axs1[1][3].set_ylabel('Oscillatory Mode dynamics', fontsize=10, fontproperties=times_font)
        axs1[1][1].set_ylim(-40, 40)
        axs1[1][1].set_xticks(range(0, 12, 1))
        axs1[1][1].set_xticklabels(['4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00'], rotation=45)
        axs1[1][2].set_ylim(-40, 40)
        axs1[1][2].set_xticks(range(0, 12, 1))
        axs1[1][2].set_xticklabels(['4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00'], rotation=45)
        axs1[1][3].set_ylim(-40, 40)
        axs1[1][3].set_xticks(range(0, 12, 1))
        axs1[1][3].set_xticklabels(['4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00', '4:00', '8:00', '13:00', '19:00'], rotation=45)
        for jj, ax in enumerate(axs1.reshape(-1)):
            ax.xaxis.set_tick_params(which='both', size=3, width=1.5, direction='out', labelsize=8)
            ax.yaxis.set_tick_params(which='both', size=3, width=1.5, direction='out', labelsize=8)
        plt.tight_layout() 
        k1 = k1 + 1
    
    # fig1.delaxes(axs1[0, 10])  
    fig1.text(0.5, 0.04, "Time", ha="center", va="center")
    plt.tight_layout(pad=3.0)  # 2-point white border

    # Save as TIFF, RGB, 400 dpi
    output_filename = "Figure 2.eps"
    #fig1.savefig(output_filename, format='eps', dpi=600)
    #plt.close(fig1)

    return col_ind1, col_ind2


def pheno_recons_fig(label1, label2, Pheno1, Pheno2, recon1, recon2, cd1, cd2, recon_modal1, recon_modal2,
                cd_modes1, cd_modes2, col_ind1, col_ind2):
    """
    Generate and save phenotype reconstruction plots with Koopman and modal reconstructions.

    Parameters:
        - label, label2: Strings for titles and file naming.
        - Pheno1, Pheno2: Phenotype data
        - recon1, recon2: Koopman reconstructions.
        - cd1, cd2: Coefficient of determination for Koopman matrix.
        - recon_modal1, recon_modal2: Modal reconstructions.
        - cd_modes1, cd_modes2: Coefficient of determination for modes.
        - col_ind1, col_ind2: reordered indices for eigenvalues and eigenmodes
    """
    x = range(0, 12, 1)
    labels = ['Malic acid\n (z-score)', u'H\u2082O\n (z-score)', u'CO\u2082\n (z-score)', 'PCK1\n (z-score)', 'PPC1\n (z-score)', 'PHO1\n (z-score)', 'PPD\n (z-score)']
    colors = ['turquoise', 'lightcoral', 'firebrick', 'lightgreen', 'yellowgreen', 'crimson',
            'palevioletred', 'steelblue', 'red', 'green', 'orchid']

    species_color1 = "#008080"
    species_color2 = "gold"
    label1 = r'$\it{C.\ major}$'
    label2 = r'$\it{C.\ rosea}$'
    fig1, axs1 = plt.subplots(len(labels), 2, figsize=(8, 15), sharey='row', sharex=True)
    fig1.patch.set_facecolor('white')

    for i, label_name in enumerate(labels):
        y0 = Pheno1.T[i, :12]
        y1 = recon1[i, :]
        y2 = Pheno2.T[i, :12]
        y3 = recon2[i, :]

        axs1[i][0].plot(x, y0, label="original distribution")
        axs1[i][0].plot(x, y1, label=f"Koopman: {round(cd1[i], 2)} R\u00b2", color=species_color1)
        axs1[i][1].plot(x, y2, label="original distribution")
        axs1[i][1].plot(x, y3, label=f"Koopman: {round(cd2[i], 2)} R\u00b2", color=species_color2)

        axs1[i][0].set_ylabel(label_name, fontsize=10, fontproperties=times_font)
        axs1[i][0].set_xticks(range(0, 12, 1))
        axs1[i][0].set_xticklabels(['4:00', '8:00', '13:00', '19:00'] * 3, fontproperties=times_font, fontsize=8, rotation=45)
        axs1[i][0].legend(loc='best', prop={'size': 8})
        
        safe_col_ind = [j for j in col_ind1 if j < len(recon_modal1)]
        recon_modal1 = np.array(recon_modal1)[safe_col_ind]
        cd_modes1 = np.array(cd_modes1)[safe_col_ind]
        CO2_ind1 = []
        for j in range(cd_modes1.shape[0]):
            if cd_modes1[j, 2] > 0.3:
                CO2_ind1.append(j)

        safe_col_ind2 = [j for j in col_ind2 if j < len(recon_modal2)]
        recon_modal2 = np.array(recon_modal2)[safe_col_ind2]
        cd_modes2 = np.array(cd_modes2)[safe_col_ind2]
        CO2_ind2 = []
        for j in range(cd_modes2.shape[0]):
            if cd_modes2[j, 2] > 0.3:
                CO2_ind2.append(j)
                
        for j, modal_recon in enumerate(recon_modal1):
            y_1 = modal_recon[i, :]
            if round(cd_modes1[j][i], 2) > 0.2:
                axs1[i][0].plot(x, y_1, label=f"Mode {j + 1}: {round(cd_modes1[j][i], 2)} R\u00b2", linewidth=1.5, color=colors[j])
        
        for j, modal_recon in enumerate(recon_modal2):
            y_2 = modal_recon[i, :]
            if round(cd_modes2[j][i], 2) > 0.2:
                axs1[i][1].plot(x, y_2, label=f"Mode {j + 1}: {round(cd_modes2[j][i], 2)} R\u00b2", linewidth=1.5, color=colors[j])
        
        axs1[i][1].set_xticks(range(0, 12, 1))
        axs1[i][1].set_xticklabels(['4:00', '8:00', '13:00', '19:00'] * 3, fontproperties=times_font, fontsize=8, rotation=45)
        axs1[i][0].legend(loc='best', prop={'size': 6})
        axs1[i][1].legend(loc='best', prop={'size': 6})

    for ax in axs1.reshape(-1):
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(times_font)

    fig1.text(0.3, 0.96, f'{label1}', ha='center', va='top', fontsize=12, weight='bold', fontproperties=times_font)
    fig1.text(0.75, 0.96, f'{label2}', ha='center', va='top', fontsize=12, weight='bold', fontproperties=times_font)
    fig1.text(0.3, 0.01, "Time", ha="center", va="bottom", fontsize=12, fontproperties=times_font)
    fig1.text(0.75, 0.01, "Time", ha="center", va="bottom", fontsize=12, fontproperties=times_font)	
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])

    output_filename = f"Figure 3.eps"
   # fig1.savefig(output_filename, format='eps', dpi=600)
   # plt.close(fig1)

    return CO2_ind1, CO2_ind2


    
def plot_species_barplot(df_species, ax, species_name, cmap, norm):
    df_species = df_species[df_species['Feature'] != 'NA']
    df_species['Mean_HSV'] = df_species.groupby('Feature')['hsv_value'].transform('mean')
    feature_counts = df_species.groupby('Feature').size().reset_index(name='Frequency')
    df_feature = pd.merge(df_species.drop_duplicates(subset='Feature')[['Feature', 'Mean_HSV']], feature_counts, on='Feature')
    df_sorted = df_feature.sort_values('Frequency', ascending=False)
    df_sorted['Feature'] = df_sorted['Feature'].str.split(' ').str[0] + '_' + df_sorted['Feature'].str.extract(r'::(H[12])')[0] + '_' + df_sorted['Feature'].str.extract(r'::(H)([1-2])_(t|p)')[2]
    colors = 'blue' # [cmap(norm(hsv)) for hsv in df_sorted['Mean_HSV']]
    bar_width = 2 
    x_positions = np.arange(len(df_sorted)) * 100  
    ax.bar(x_positions, df_sorted['Frequency'], color=colors, alpha=0.85, width=bar_width)
    ax.set_xlabel("Features", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_sorted['Feature'], rotation=90, ha="center", fontsize=10, color='black') 
    ax.set_title(f"{species_name}", fontsize=18)
    
   