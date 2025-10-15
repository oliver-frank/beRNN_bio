import os
import nibabel as nib
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, masking, image
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker # from nilearn.maskers import NiftiMapsMasker

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, linregress

# Ensure Matplotlib is not in interactive mode
plt.ioff()


########################################################################################################################
# Correlation Matrix
########################################################################################################################
participants = ['sub-SNIP6IECX', 'sub-SNIP96WID', 'sub-SNIPKPB84', 'sub-SNIPYL4AS'] # 'sub-SNIPDKHPB'
recordings = ['05'] # ,
modes = ['SCHAEFER'] # , 'AAL', 'MSDL'
n_rois = 200
tasks = ['faces', 'flanker', 'nback', 'rest', 'reward'] #  is for 03 MRI not working, let's wait for preprocessing 04 if same failure occurs
data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\bio_BeRNN'
directory = 'W:\\group_csp\\analyses\\oliver.frank\\brainModels'
threshold = 0.5  # threshold defining sparsity in created graph

# Load average_correlationMatrixList
def apply_threshold(matrix, threshold):
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix, 0)
    return matrix_thresholded

for participant in participants: # one participant after the other
    for recording in recordings: # one recording at a time
        subject_directory = os.path.join(directory, f'{participant + recording}', 'func')

        for mode in modes:
            correlationMatrixList = []
            correlationMatrixNameList = []
            for task in tasks: # average over all tasks of one recording

                nifti_file = os.path.join(subject_directory, f'{participant+recording}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')
                # taskList.append(nifti_file)
                # Fetch the right atlas
                if mode == 'AAL':
                    aal_atlas = datasets.fetch_atlas_aal(data_dir=data_dir)
                    atlas_filename = aal_atlas.maps
                    labels = aal_atlas.labels
                elif mode == 'MSDL':
                    msdl_atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)
                    atlas_filename = msdl_atlas.maps
                    labels = msdl_atlas.labels
                elif mode == 'SCHAEFER':
                    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(data_dir=data_dir, n_rois=n_rois, yeo_networks=7)
                    atlas_filename = schaefer_atlas.maps
                    labels = schaefer_atlas.labels  # These are the names of the regions

                fmri_img = nib.load(nifti_file)

                # Step 3: Use the chosen atlas to extract parcelled activitiy regions from high-resolution fMRI data
                if mode == 'AAl' or mode == 'SCHAEFER':
                    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, memory='nilearn_cache', detrend=True)
                else:
                    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache')

                time_series = masker.fit_transform(fmri_img)

                # Step 4: Compute the correlation matrix
                correlation_matrix = np.corrcoef(time_series.T)
                correlation_matrix_name = f'{participant+recording}' + '_' + task
                correlationMatrixList.append(correlation_matrix)
                # Save the correlation matrix as a .npy file
                corrMatrixPath = os.path.join(subject_directory,f'npy_corrMatrices_{mode}_{n_rois}_{participant+recording}')

                if not os.path.exists(corrMatrixPath):
                    # If it doesn't exist, create the directory
                    os.makedirs(corrMatrixPath)
                    print(f"Directory created: {corrMatrixPath}")
                else:
                    print(f"Directory already exists: {corrMatrixPath}")

                np.save(os.path.join(corrMatrixPath, correlation_matrix_name), correlation_matrix)

                # Step 5: Plot the task correlation matrix
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Customize as needed
                cax  = plt.imshow(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
                plt.colorbar(cax, ax=ax, label='Correlation coefficient')

                # Set title and labels
                ax.set_title(f'Correlation Matrix ({mode} Atlas) - {participant + recording} - {task} - {n_rois}', fontsize=14)
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=90, fontsize=6)
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels, fontsize=6)

                # Adjust layout and render the plot
                plt.tight_layout()
                # plt.draw()  # Ensure all elements are rendered before saving

                # Save figure
                fname = f'Correlation Matrix ({mode} Atlas) - {participant + recording} - {task} - {n_rois}' + '.png'
                fpath = f'{directory}\\visuals\\Correlation_fMRI\\{mode}\\{participant + recording}'

                if not os.path.exists(fpath):
                    # If it doesn't exist, create the directory
                    os.makedirs(fpath)
                    print(f"Directory created: {fpath}")
                else:
                    print(f"Directory already exists: {fpath}")

                plt.savefig(os.path.join(fpath, fname), format='png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close(fig)

            averaged_correlation_matrix = np.mean(correlationMatrixList, axis=0)
            averaged_correlation_matrix_name = f'{participant+recording}' + '_' + f'averagedTask - {n_rois}'

            np.save(os.path.join(corrMatrixPath, averaged_correlation_matrix_name), averaged_correlation_matrix)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Customize as needed
            cax = plt.imshow(averaged_correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
            plt.colorbar(cax, ax=ax, label='Correlation coefficient')

            # Set title and labels
            ax.set_title(f'Correlation Matrix ({mode} Atlas) - {participant + recording} - averaged tasks - {n_rois}', fontsize=14)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels, fontsize=6)

            plt.tight_layout()
            # plt.draw()

            # Save figure
            fname = f'Correlation Matrix ({mode} Atlas) - {participant + recording} - averaged tasks - {n_rois}' + '.png'
            plt.savefig(os.path.join(fpath, fname), format='png', dpi=300, bbox_inches='tight')

            plt.show()
            plt.close(fig)


            ########################################################################################################################
            # Topological Marker Analysis - Brain
            ########################################################################################################################
            # for idx, average_correlationMatrix in enumerate(average_correlationMatrixList):
            # Define a threshold (you can experiment with this value)
            averaged_correlation_matrix_thresholded = apply_threshold(averaged_correlation_matrix, threshold)

            # Function to apply a threshold to the matrix
            G = nx.from_numpy_array(averaged_correlation_matrix_thresholded)

            # The degree of a node in a graph is the count of edges connected to that node. For each node, it represents the number of
            # direct connections (or neighbors) it has within the graph.
            degrees = nx.degree(G) # For calculating node degrees. attention: transform into graph then apply stuff
            # Betweenness centrality quantifies the importance of a node based on its position within the shortest paths between other nodes.
            betweenness = nx.betweenness_centrality(G) # For betweenness centrality.
            # Closeness centrality measures the average distance from a node to all other nodes in the network.
            closeness = nx.closeness_centrality(G) # For closeness centrality.
            # average_path_length = nx.average_shortest_path_length(G) # For average path length. attention: Graph is disconnected after thresholding therefore not possible

            # Optionally calculate averages of node-based metrics
            avg_degree = np.mean(list(dict(G.degree()).values()))
            avg_betweenness = np.mean(list(betweenness.values()))
            avg_closeness = np.mean(list(closeness.values()))

            # Graph Density: Measures how many edges exist compared to the maximum possible. This can give you a sense of how interconnected the network is.
            # Max 1; Min 0 Everything or Nothing is connected
            density = nx.density(G)
            # Assortativity: Measures the tendency of nodes to connect to other nodes with similar degrees.
            # A positive value means that high-degree nodes tend to connect to other high-degree nodes. Around 0 no relevant correlation
            assortativity = nx.degree_assortativity_coefficient(G)
            # Transitivity (Global Clustering Coefficient): Measures the likelihood that the adjacent nodes of a node are connected.
            # It’s an indicator of local clustering. 0-1 ; 1 every node that has two neighbours are also connected to each other
            transitivity = nx.transitivity(G)
            # Average Clustering Coefficient: Provides the average of the clustering coefficient (local) for all nodes.
            # It gives you a sense of how well nodes tend to form clusters. Similar to Transitivity
            avg_clustering = nx.average_clustering(G)
            # Largest Connected Component: The size (number of nodes) in the largest connected subgraph of the network.
            # This is particularly important in sparse graphs after thresholding.
            largest_cc = len(max(nx.connected_components(G), key=len))

            metrics = {
                "degrees": degrees,
                "betweenness": betweenness,
                "closeness": closeness,
                "avg_degree": avg_degree,
                "avg_betweenness": avg_betweenness,
                "avg_closeness": avg_closeness,
                "density": density,
                "assortativity": assortativity,
                "transitivity": transitivity,
                "avg_clustering": avg_clustering,
                "largest_cc": largest_cc
            }

            # Define the output directory and file name
            output_directory = os.path.join(directory,'topologicalMarkers_threshold_' + str(threshold))

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Save the metrics dictionary as a .npy file
            output_file = os.path.join(output_directory, f'topologicalMarkers_threshold_{threshold}_{mode}_{n_rois}_{participant + recording}.npy')
            np.save(output_file, metrics, allow_pickle=True)

            print(f"Network measures saved to: {output_file}")



# ########################################################################################################################
# # Analysis of Topological Marker change over time
# ########################################################################################################################
# # Load several dictionaries, sort them in a df and compare them statstically
# metricsPath = "W:\\group_csp\\analyses\\oliver.frank\\brainModels\\topologicalMarkers\\SCHAEFER" # fix: Several metricsPath needed
# metricsDirectoryList_all = os.listdir(metricsPath)
#
# participants = ['SNIP96WID'] # , 'SNIPKPB84', 'SNIPYL4AS' 'SNIP6IECX', 'SNIP96WID'
# participants_beRNN = ['BeRNN_05'] # 'BeRNN_03', ,
# # recordingsList = ['01', '02', '03', '04', '05']
# recordingsList = ['04']
#
# for number_beRNN, participant in enumerate(participants):
#     metricsList = []
#     for i in recordingsList:
#         metrics = np.load(os.path.join(metricsPath, 'sub-'+participant+i, f'threshold_{threshold}', f'topologicalMarkers_SCHAEFER_sub-{participant+i}_threshold_{threshold}.npy'), allow_pickle=True).item()
#         metricsList.append(metrics)
#
#     df = pd.DataFrame(metricsList)
#
#     # Define time points
#     time = [1, 2, 3, 4]
#
#     # Create a plot with subplots for each metric between columns 3 and 11
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))  # Adjust grid based on number of metrics
#     fig.suptitle(f'Linear Regression for each metric over time - {participants_beRNN[number_beRNN]}', fontsize=16)
#
#     # Flatten axes array for easier indexing
#     axes = axes.flatten()
#
#     # Loop through each metric column (3 to 11)
#     for idx, col in enumerate(range(3, 11)):  # Adjust range as needed
#         values = df.iloc[:, col]
#
#         # Perform linear regression
#         slope, intercept, r_value, p_value, std_err = linregress(time, values)
#         regression_line = [slope * t + intercept for t in time]
#
#         # Plot data points and regression line on the respective subplot
#         axes[idx].scatter(time, values, color='blue', label='Data Points')
#         axes[idx].plot(time, regression_line, color='red', label=f'Regression Line (slope={slope:.2f})')
#         axes[idx].set_title(f'Metric: {df.columns[col]}')
#         axes[idx].set_xlabel('Time')
#         axes[idx].set_ylabel('Value')
#         axes[idx].legend()
#
#     # Adjust layout and show plot
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Rect option to fit main title
#
#     figure_directory = f'W:\\group_csp\\analyses\\oliver.frank\\brainModels\\topologicalMarkers\\SCHAEFER\\trends\\{threshold}'
#     if not os.path.exists(figure_directory):
#         os.makedirs(figure_directory)
#
#     plt.savefig(os.path.join(figure_directory, f'topologicalMarkerTrends_{participants_beRNN[number_beRNN]}.png'), format='png', dpi=300, bbox_inches='tight')
#
#     plt.show()
#     plt.close()



# ########################################################################################################################
# # Correlation Matrices of Correlation Matrices
# ########################################################################################################################
# # Functional correlation matrices - Pearson Correlation
# import numpy as np
# import matplotlib.pyplot as plt
# import itertools
# from scipy.stats import pearsonr
#
# for participant in participants:
#     mode = '200' # info: filtered_rois, 200, 400, ' '
#     # File paths
#     files = [
#         # f"W:\\group_csp\\analyses\\oliver.frank\\brainModels\\sub-{participant}01\\func\\npy_corrMatrices_SCHAEFER_{mode}_sub-{participant}01\\sub-{participant}01_averagedTask - {mode}.npy",
#         # f"W:\\group_csp\\analyses\\oliver.frank\\brainModels\\sub-{participant}02\\func\\npy_corrMatrices_SCHAEFER_{mode}_sub-{participant}02\\sub-{participant}02_averagedTask - {mode}.npy",
#         # f"W:\\group_csp\\analyses\\oliver.frank\\brainModels\\sub-{participant}03\\func\\npy_corrMatrices_SCHAEFER_{mode}_sub-{participant}03\\sub-{participant}03_averagedTask - {mode}.npy",
#         f"W:\\group_csp\\analyses\\oliver.frank\\brainModels\\sub-{participant}04\\func\\npy_corrMatrices_SCHAEFER_{mode}_sub-{participant}04\\sub-{participant}04_averagedTask - {mode}.npy",
#         # f"W:\\group_csp\\analyses\\oliver.frank\\brainModels\\sub-{participant}05\\func\\npy_corrMatrices_SCHAEFER_{mode}_sub-{participant}05\\sub-{participant}05_averagedTask - {mode}.npy"
#     ]
#
#     # Load matrices
#     matrices = [np.load(file) for file in files]
#
#     # Function to extract the upper triangle (excluding the diagonal)
#     def upper_triangle(matrix):
#         return matrix[np.triu_indices_from(matrix, k=1)]
#
#     # Extract the upper triangles of all matrices
#     flattened_matrices = [upper_triangle(mat) for mat in matrices]
#
#     # Calculate pairwise correlations
#     pairwise_correlations = {}
#     for (i, vec1), (j, vec2) in itertools.combinations(enumerate(flattened_matrices), 2):
#         corr, _ = pearsonr(vec1, vec2)  # Pearson correlation
#         pairwise_correlations[f"{files[i]} vs {files[j]}"] = corr
#
#     # Display pairwise correlations
#     print("Pairwise Correlation of Correlations:")
#     for pair, corr in pairwise_correlations.items():
#         print(f"{pair}: Correlation = {corr:.4f}")
#
#     # Visualize correlation matrices
#     for i, matrix in enumerate(matrices):
#         plt.figure()
#         plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
#         plt.colorbar(label="Correlation")
#         plt.title(f"Correlation Matrix: {files[i]}")
#         plt.xlabel("Region")
#         plt.ylabel("Region")
#         plt.show()


