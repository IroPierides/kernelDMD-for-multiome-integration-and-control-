# kernelDMD-for-multiome-integration-and-control-

This is code for reproducing the results in the paper 'kernel DMD for multiome integration and control' (Pierides I., Kramml M. H, Waldherr S. and Weckwerth W.) All was coded using Python 3.13.7 with the following dependencies: pandas 2.3.3, numpy 2.2.6, scipy 1.16.2, cvxpy 1.6.0, scikit-learn 1.7.2, joblib 1.5.2, matplotlib 3.9.4, networkx 2.5, seaborn 0.13.2. All system dependencies are included in the myenv.yml file. 

The Jupyter notebook file (phenocopying_biological_systems.ipynb) is the main file for running the analysis and includes system identification (Koopman operator via kernel DMD) and system control (after data reduction and alignment). The idea is to identify dynamic differences between two Clusia species via eigenmode analysis and then shift output distributions of one species to that of the other. This identifies important control features that control the different modes of photosynthesis.

The python files include important functions required in the main sript. The figures.py files includes functions for some of the plots, the preprocessing.py file includes functions important for preprocessing of the data such as timepoint rearrangement and scaling and the reconstructions.py file includes functions for data and output reconstruction accuracies. The SystemIdentification_and_Control.py file containts functions required for model identification and control such as the kernel function or linearMPC. 

To run the script in an sbatch make the following sbatch script: 

 

#SBATCH --cpus-per-task=32
#SBATCH --mem=900000
#SBATCH -t 00:10:00

module load Conda
conda env create --prefix /path/to/files/env -f /path/to/files/myenv.yml
conda activate /path/to/files/env

pip install jupyter
jupyter nbconvert --to notebook --execute --inplace phenocopying_biological_systems.ipynb

conda deactivate
