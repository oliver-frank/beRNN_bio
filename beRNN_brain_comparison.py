import numpy as np
# from pathlib import Path
import scipy.stats
import sys
import os
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.linalg import orthogonal_procrustes
from scipy.stats import ttest_rel
from scipy.stats import spearmanr
# from sklearn.manifold import MDS
from pathlib import Path
from numpy import arctanh
# from scipy.stats import ttest_ind, ks_2samp
# from collections import OrderedDict
# from network import Model
# import tools
# import pickle
# import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

def plot_rsa(directory_beRNN, directory_brain, participantList, rdm_taskset, folder, currentFileEnding, dataType):
    def ascendingNumbers_beRNN(e):
        return int(e.split('_')[0])

    def ascendingNumbers_brain(e):
        return int(e.split('brainImage0')[1].split('_')[0])

    def vec(rdm):  # → length 66 (for 12 tasks)
        idx = np.triu_indices_from(rdm, k=1)
        return rdm[idx]

    rsa_directory = Path(*directory_brain.parts[:-1], '_networkComparison_beRNN_brain')
    os.makedirs(rsa_directory, exist_ok=True)


    # Gather rdmFiles in dict lists
    rdm_dict_beRNN = {}
    for participant in participantList:
        directory_ = Path(directory_beRNN, f'{participant}/visuals/performance_test/{rdm_taskset}')

        # Check if all defined participant in list were preprocessed - exit function if not
        if not directory_.exists():
            print(f"Directory {directory_} does not exist. Exiting.")
            sys.exit()

        rdmFiles = [i for i in os.listdir(str(directory_).format(participant=participant)) if i.endswith('.npy')]
        rdmFiles.sort(key=ascendingNumbers_beRNN)  # Sort list according to information chunk given in key function
        # info: Only have 3 brainImages from beRNN_04
        if participant == 'beRNN_04': numberOfModels = 3
        else: numberOfModels = 5

        rdm_dict_beRNN[participant] = rdmFiles[0:numberOfModels] # Only take defied number of models to be aligned with brain rdms

    # Load all .npy files as ndarrays and save them in already existing dict
    for participant in participantList:
        # info: Only have 3 brainImages from beRNN_04
        if participant == 'beRNN_04': numberOfModels = 3
        else: numberOfModels = 5

        for rdm in range(0,numberOfModels):
            rdm_dict_beRNN[participant][rdm] = np.load(Path(directory_beRNN, f'{participant}/visuals/performance_test/{rdm_taskset}', rdm_dict_beRNN[participant][rdm]))


    rdm_dict_brain = {}
    for brain in participantList:
        # Path(f'C:/Users/oliver.frank/Desktop/PyProjects/brainModels/rdMatrices_brain_{representationMetric}')
        # directory_ = Path(*directory_brain, f'{brain}_brainImage05_rdMatrix_varNorm_noReward')

        # Check if all defined participant in list were preprocessed - exit function if not
        if not directory_brain.exists():
            print(f"Directory {directory_brain} does not exist. Exiting.")
            sys.exit()

        # rdmFiles = [i for i in os.listdir(str(directory_).format(brain=brain)) if i.endswith('.npy')]
        rdmFiles = [i for i in os.listdir(directory_brain) if brain in i and i.endswith('.npy')]
        rdmFiles.sort(key=ascendingNumbers_brain)  # Sort list according to information chunk given in key function
        rdm_dict_brain[brain] = rdmFiles

    # Load all .npy files as ndarrays and save them in already existing dict
    for brain in participantList:
        # info: Only have 3 brainImages from beRNN_04
        if brain == 'beRNN_04': numberOfModels = 3
        else: numberOfModels = 5

        for rdm in range(0,numberOfModels):
            rdm_dict_brain[brain][rdm] = np.load(Path(directory_brain, rdm_dict_brain[brain][rdm]))



    # # Align rdmLists to same length
    # min_length = min([len(rdm_dict[rdmList]) for rdmList in rdm_dict])
    # for participant in participantList:
    #     rdm_dict[participant] = rdm_dict[participant][:min_length]
    # # Load the rdm files
    # for participant in participantList:
    #     # Re-allocate the right participant-specific path
    #     directory_ = Path(*directory.parts[:-1],f'{participant}/visuals/performance_test/representationalDissimilarity_cosine')
    #     for rdm in range(min_length):
    #         rdm_dict[participant][rdm] = np.load(os.path.join(str(directory_).format(participant=participant), rdm_dict[participant][rdm]))


    # Create vectors from rdm ndarrays for upper triangle of symmetric rdms
    rdm_vec_dict_beRNN = {s: [vec(r) for r in rdm_list] for s, rdm_list in rdm_dict_beRNN.items()}
    rdm_vec_dict_brain = {s: [vec(r) for r in rdm_list] for s, rdm_list in rdm_dict_brain.items()}

    # Compute spearman rank correlation for each possible combination of model groups (within/between)
    subjects = list(rdm_vec_dict_beRNN.keys()) # fix should be the same
    # subjects = list(rdm_vec_dict_brain.keys()) # fix should be the same
    # N_subj = len(subjects)

    within_rsa = {s: [] for s in subjects}
    between_rsa = {s: [] for s in subjects}

    # Loop over all combinations and assign results to the right dict (within/between)
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            for v1 in rdm_vec_dict_beRNN[s1]:
                for v2 in rdm_vec_dict_brain[s2]:
                    rho = scipy.stats.spearmanr(v1, v2).correlation
                    if s1 == s2:
                        within_rsa[s1].append(rho)
                    else:
                        between_rsa[s1].append(rho)

    # Create dict for each comparison's mean value
    within_mean = np.array([np.mean(within_rsa[s]) for s in subjects])  # shape (5,)
    between_mean = np.array([np.mean(between_rsa[s]) for s in subjects])  # shape (5,)

    z_within = arctanh(within_mean)  # Fisher z
    z_between = arctanh(between_mean)

    t, p = ttest_rel(z_within, z_between)  # H₀: means equal
    print(f'Within  mean ρ = {within_mean.mean():.3f}')
    print(f'Between mean ρ = {between_mean.mean():.3f}')
    print(f'Paired t(4) = {t:.2f}, p = {p:.4f}')

    # Create ordered list of vectors and labels
    all_vecs_beRNN = []
    labels_beRNN = []
    for subj in subjects:
        for i, vec in enumerate(rdm_vec_dict_beRNN[subj], 1):
            all_vecs_beRNN.append(vec)
            labels_beRNN.append(f"{subj}_M{i}")

    # Create ordered list of vectors and labels
    all_vecs_brain = []
    labels_brain = []
    for subj in subjects:
        for i, vec in enumerate(rdm_vec_dict_brain[subj], 1):
            all_vecs_brain.append(vec)
            labels_brain.append(f"brain_{subj.split('_')[1]}_M{i}")

    # Compute full Spearman RSA similarity matrix
    n_models_beRNN = len(all_vecs_beRNN)
    n_models_brain = len(all_vecs_brain)
    rsa_sim = np.zeros((n_models_beRNN, n_models_brain))
    for i in range(n_models_beRNN):
        for j in range(n_models_brain):
            rho, _ = spearmanr(all_vecs_beRNN[i], all_vecs_brain[j])
            rsa_sim[i, j] = rho

    # Convert to dissimilarity
    rsa_dissim = 1 - rsa_sim

    # Plot RSA heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        rsa_dissim,
        cmap="vlag",
        xticklabels=labels_beRNN,
        yticklabels=labels_brain,
        square=True,
        linewidths=0.1,
        cbar_kws={
            'shrink': 0.5,
            'aspect': 10,
            'label': 'RDA (1 - Spearman ρ)',
            'ticks': np.linspace(0, 1, 6)
        },
        vmin=0, vmax=1
    )

    plt.title("RDA matrix between all (66_upTri) RDM model representations - beRNN/brain", fontsize=10)
    plt.xlabel("Model beRNN", fontsize=8)
    plt.ylabel("Model brain", fontsize=8)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    # plt.show()
    plt.savefig(os.path.join(rsa_directory, f'RDAmatrix_beRNN-{folder}-{dataType}_brain-{currentFileEnding}.png'))

# info: The beRNNs need to be pre-processed by beRNN_v1/hyperparameterOverview.py and beRNN_v1/taskRepresentation.py
# This script computes rdm for brain models and then optionally compares brain with beRNN rdm models in final rsa

# Participants
participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
# participants_brain_models = ['SNIPKPB84', 'SNIPYL4AS', 'SNIP6IECX', 'SNIP96WID']
participants_brain_models = ['SNIPDKHPB']
# numberOfModels = 5 # number of brain models per participant
numberOfModels = 3 # number of brain models per participant

directory = r'C:\Users\oliver.frank\Desktop\PyProjects\brainModels'
os.makedirs(os.path.join(directory, f'rdMatrices_brain'), exist_ok=True)

folder = 'robustnessTest_allParticipants_softplus4win_fundamentals'
dataType = 'highDim'

directory_beRNN = Path(f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/{folder}/{dataType}')
directory_brain = Path(f'C:/Users/oliver.frank/Desktop/PyProjects/brainModels/rdMatrices_brain')

rdm_taskset = 'representationalDissimilarity_cosine_fundamentals'

beRNN_brain_dict = {
    'SNIPKPB84': 'beRNN_01',
    'SNIPYL4AS': 'beRNN_02',
    'SNIP6IECX': 'beRNN_03',
    'SNIPDKHPB': 'beRNN_04',
    'SNIP96WID': 'beRNN_05'
}

# ruleset = 'all'
# rules = ['faces', 'flanker', 'nback', 'rest', 'reward']

# head: Compute Representational Dissimilarity Matrix (RSA-style) ==========================================
# info: Just preprocess the brain .npy files here after weightMatrices_dti_v2.py - beRNN should have already be preprocessed in beRNN_v1

# info: File ending of preprocessed dti matrices from weightMatrices_dti_v2.py
currentFileEnding = 'relevantRois_taskMatrix_varZ.npy'

# for participant_brain_model in participants_brain_models:
#     for modelNumber in range(1,numberOfModels+1):
#         varNorm_taskMatrix = np.load(os.path.join(directory, f'sub-{participant_brain_model}0{modelNumber}', \
#                              'func', 'allTask_representationMatrices', f'sub-{participant_brain_model}0{modelNumber}{currentFileEnding}'))
#
#         task_matrix = varNorm_taskMatrix  # shape needed: (n_tasks, n_units)
#         rdm_metric = 'cosine'  # correlation - cosine - ...
#         rdm = squareform(pdist(task_matrix, metric=rdm_metric))
#         rdm_vector = rdm[np.triu_indices_from(rdm, k=1)]
#
#         participantID = beRNN_brain_dict[participant_brain_model]
#         np.save(fr'{directory}\rdMatrices_brain\{participantID}_brainImage0{modelNumber}_rdMatrix{currentFileEnding}', rdm) # info: in beRNN analysis plot_rsa is also just based on matrix not vector
#         # np.save(fr'C:\Users\oliver.frank\Desktop\PyProjects\brainModels\rdMatrices_brain\sub-{participant_brain_model}0{modelNumber}_rdVector', rdm_vector) # ?
#

plot_rsa(directory_beRNN, directory_brain, participantList, rdm_taskset, folder, currentFileEnding, dataType)

















# # Topological markers
# top_markers = ["degree", "betweenness", "assortativity"]
#
# # Months for Model 1 (Model 2 has no months)
# months = ["3", "4", "5"]
#
# # Output directory
# destination_dir = "W:\\group_csp\\analyses\\oliver.frank\\brainModels\\topMarkerComparisons_beRNN_brain"
# os.makedirs(destination_dir, exist_ok=True)
#
# def load_bootstrap_distributions(directory, participant, top_marker, months=months):
#     """
#     Load bootstrap distributions for a given participant and topological marker.
#     """
#     distributions = {}
#
#     if 'beRNNmodels' in directory:
#         # beRNN model (Has months)
#         for month in months:
#             file_path = os.path.join(directory, f"{top_marker}List_{month}.npy")
#             if os.path.exists(file_path):
#                 distributions[month] = np.load(file_path, allow_pickle=True)
#             else:
#                 print(f"Missing file: {file_path}")
#
#     else:
#         # brain model (No months, so duplicate data across months)
#         file_path = os.path.join(directory, f"{participant}_bootstrap_{top_marker}.npy")
#         if os.path.exists(file_path):
#             data = np.load(file_path, allow_pickle=True)
#             distributions = {month: data for month in months}  # Replicate for 3 months
#         else:
#             print(f"Missing file: {file_path}")
#
#     return distributions
#
# def compare_models(beRNN_model_distributions, brain_model_distributions, participant_indice):
#     """
#     Compare Model Class 1 and Model Class 2 distributions for each topological marker across months.
#     """
#     p_values = {marker: {} for marker in top_markers}
#
#     for marker in top_markers:
#         for month in months:
#             if month in beRNN_model_distributions[marker] and month in brain_model_distributions[marker]:
#                 data_model1 = beRNN_model_distributions[marker][month]
#                 data_model2 = brain_model_distributions[marker][month]
#
#                 if len(data_model1) > 1 and len(data_model2) > 1:
#                     t_stat, p_ttest = ttest_ind(data_model1, data_model2, equal_var=False)
#                     ks_stat, p_ks = ks_2samp(data_model1, data_model2)
#
#                     p_values[marker][month] = min(p_ttest, p_ks)  # Store the smaller p-value
#                 else:
#                     p_values[marker][month] = 1.0  # No valid comparison
#
#     # Convert to DataFrame for visualization
#     p_df = pd.DataFrame(p_values).T  # Transpose so markers are rows, months are columns
#
#     # Prepare text annotations with significance levels
#     def format_p_value(p):
#         if p < 0.001:
#             return f"$\\bf{{{p:.3f}}}$***"  # Bold + ***
#         elif p < 0.01:
#             return f"$\\bf{{{p:.3f}}}$**"   # Bold + **
#         elif p < 0.05:
#             return f"$\\bf{{{p:.3f}}}$*"    # Bold + *
#         else:
#             return f"{p:.3f}"               # No bold
#
#     annotations = p_df.applymap(format_p_value)
#
#     # Plot heatmap of p-values
#     plt.figure(figsize=(10, 6))
#     ax = sns.heatmap(
#         p_df.astype(float),
#         annot=annotations,
#         fmt="",
#         cmap="magma",
#         vmin=0.001,
#         vmax=1.0,
#         center=0.05,
#         cbar_kws={"shrink": 1.0},
#         annot_kws={"fontsize": 10, "color": "white"},
#     )
#
#     plt.title(f"Statistical Comparison: beRNN_model vs brain_model - {participants_beRNN_models[participant_indice]}")
#     plt.xlabel("Months")
#     plt.ylabel("Topological Markers")
#
#     # Save and show the plot
#     plot_path = os.path.join(destination_dir, f"topMarkerComparison_{participants_beRNN_models[participant_indice]}_beRNN_model_vs_brain_model.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.show()
#
# # Process each participant
# for participant_indice in range(0,len(participants_beRNN_models)):
#     print(f"Processing participant: {participants_beRNN_models[participant_indice]}")
#     print(f"Processing participant: {participants_brain_models[participant_indice]}")
#
#     # Directories for the two model classes
#     beRNN_model_dir = f"C:\\Users\\oliver.frank\\Desktop\\BackUp\\beRNNmodels\\2025_03_2\\{participants_beRNN_models[participant_indice]}\\overviews\\distributions"  # 9 Distributions per participant
#     brain_model_dir = "W:\\group_csp\\analyses\\oliver.frank\\brainModels\\topologicalMarkers_threshold_0.4\\topologicalMarkers_bootstrap"  # 3 Distributions per participant
#
#     # Load distributions
#     beRNN_model_distributions = {marker: load_bootstrap_distributions(beRNN_model_dir, participants_beRNN_models[participant_indice], marker, months) for marker in top_markers}
#     brain_model_distributions = {marker: load_bootstrap_distributions(brain_model_dir, participants_brain_models[participant_indice], marker) for marker in top_markers}
#
#     # Compare and visualize
#     compare_models(beRNN_model_distributions, brain_model_distributions, participant_indice)
#
# print("All comparisons completed!")
