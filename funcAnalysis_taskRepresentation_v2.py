import os
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from sklearn.preprocessing import StandardScaler


# ------------------------------
# Parameters
# ------------------------------
participants = ['sub-SNIP6IECX', 'sub-SNIP96WID', 'sub-SNIPKPB84', 'sub-SNIPYL4AS']
# participants = ['sub-SNIPDKHPB']
recordings = ['01', '02', '03', '04', '05']
# recordings = ['01', '02', '03']
tasks = ['faces', 'flanker', 'nback', 'reward']
atlas_mode = 'SCHAEFER'  # 'AAL', 'MSDL', or 'SCHAEFER'
relevantRegions_filter = True
n_rois = 200

# data_dir = r"C:\Users\oliver.frank\Desktop\BackUp\beRNN_bio"
data_dir = r'C:\Users\oliver.frank\Desktop\PyProjects\brainModels'
atlas_dir = r"C:\Users\oliver.frank\Desktop\PyProjects\beRNN_bio\schaefer_2018"
output_folder = 'allTask_representationMatrices'


# ------------------------------
# Load Schaefer ROI labels
# ------------------------------
labels_file = os.path.join(atlas_dir, f"Schaefer2018_{n_rois}Parcels_7Networks_order.txt")
with open(labels_file, "r") as f:
    labels = [line.strip() for line in f]

roi_indices = [i for i, lab in enumerate(labels) if "Cont" in lab or "DorsAttn" in lab or "Default" in lab]


# ------------------------------
# Function to extract time series
# ------------------------------
def get_task_vectors(fmri_file, atlas_file, mode='SCHAEFER', select_rois=None):
    fmri_img = nib.load(fmri_file)

    if mode in ['AAL', 'SCHAEFER']:
        masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=False, detrend=True)
    else:
        masker = NiftiMapsMasker(maps_img=atlas_file, standardize=True)

    time_series = masker.fit_transform(fmri_img)  # shape: (timepoints, n_rois)

    if select_rois is not None:
        time_series = time_series[:, select_rois]

    # Mean vector across time
    mean_vector = time_series.mean(axis=0)
    # Variance vector across time
    var_vector = time_series.var(axis=0)

    # Z-score variance across ROIs to make comparable across tasks
    var_vector_z = StandardScaler().fit_transform(var_vector.reshape(-1, 1)).ravel()

    return mean_vector, var_vector_z


# ------------------------------
# Main loop
# ------------------------------
for participant in participants:
    for recording in recordings:
        subject_dir = os.path.join(data_dir, f'{participant}{recording}', 'func')
        os.makedirs(os.path.join(subject_dir, output_folder), exist_ok=True)

        mean_matrix = []
        var_matrix = []

        for task in tasks:
            fmri_file = os.path.join(subject_dir,
                                     f'{participant}{recording}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')

            atlas_file = os.path.join(atlas_dir, f"Schaefer2018_{n_rois}Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")

            select = roi_indices if relevantRegions_filter else None
            mean_vec, var_vec = get_task_vectors(fmri_file, atlas_file, mode=atlas_mode, select_rois=select)

            mean_matrix.append(mean_vec)
            var_matrix.append(var_vec)
            print(f'Processed {task} for {participant}{recording}')

        mean_matrix = np.vstack(mean_matrix)  # shape: (n_tasks, n_rois)
        var_matrix = np.vstack(var_matrix)  # shape: (n_tasks, n_rois)

        # Save matrices
        if relevantRegions_filter == False:
            np.save(os.path.join(subject_dir, output_folder, f'{participant}{recording}_taskMatrix_mean.npy'), mean_matrix)
            np.save(os.path.join(subject_dir, output_folder, f'{participant}{recording}_taskMatrix_varZ.npy'), var_matrix)
        else:
            np.save(os.path.join(subject_dir, output_folder, f'{participant}{recording}_relevantRois_taskMatrix_mean.npy'),mean_matrix)
            np.save(os.path.join(subject_dir, output_folder, f'{participant}{recording}_relevantRois_taskMatrix_varZ.npy'),var_matrix)

        print(f'Saved task matrices for {participant}{recording}')



# import matplotlib.pyplot as plt
# import seaborn as sns
#
# var_matrix = np.load(r'C:\Users\oliver.frank\Desktop\PyProjects\brainModels\rdMatrices_brain\beRNN_05_brainImage05_rdMatrix_relevantRois_taskMatrix_varZ.npy')
# mean_matrix = np.load(r'C:\Users\oliver.frank\Desktop\PyProjects\brainModels\rdMatrices_brain\beRNN_05_brainImage05_rdMatrix_relevantRois_taskMatrix_mean.npy')
# # Visualize mean and variance matrices
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Better choice
# sns.heatmap(var_matrix, ax=axes[0], cmap='viridis')
# axes[0].set_title(f'{participant}{recording} | Variance per ROI')
# axes[0].set_xlabel('ROIs')
# axes[0].set_ylabel('Tasks')
# axes[0].set_yticks(np.arange(len(tasks)) + 0.5)
# axes[0].set_yticklabels(tasks, rotation=0)
#
# sns.heatmap(mean_matrix, ax=axes[1], cmap='viridis')
# axes[1].set_title(f'{participant}{recording} | Mean per ROI')
# axes[1].set_xlabel('ROIs')
# axes[1].set_ylabel('Tasks')
# axes[1].set_yticks(np.arange(len(tasks)) + 0.5)
# axes[1].set_yticklabels(tasks, rotation=0)
#
# plt.tight_layout()
# plt.show()


