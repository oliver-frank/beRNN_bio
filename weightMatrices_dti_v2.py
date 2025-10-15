"""
DTI -> tractography -> connectivity pipeline
Saves connectomes at original atlas resolution and downsampled 256/128/64/32, plus an upsampled version.

Dependencies:
  pip install numpy scipy scikit-learn dipy nibabel nilearn matplotlib

Run:
  update the `participant`, `base_path` and file names at the bottom and run.
"""

import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.utils import seeds_from_mask, connectivity_matrix
from dipy.tracking.streamline import Streamlines
from scipy.ndimage import gaussian_filter
from scipy.ndimage import center_of_mass
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings("ignore")

def robust_normalize(mat, percentile=99.5, clip=(0,1), do_log=True):
    """
    Robust normalization: optional log1p -> divide by a high percentile -> clip to [0,1]
    """
    m = mat.copy()
    if do_log:
        m = np.log1p(m)  # compress heavy tails while preserving rank/order
    p = np.percentile(m, percentile)
    if p == 0:
        p = m.max() if m.max() != 0 else 1.0
    m = m / p
    m = np.clip(m, clip[0], clip[1])
    return m

def ensure_int_labels(atlas_img):
    """Return atlas data as integer labels and the max label"""
    atlas_data = np.asarray(atlas_img.get_fdata())
    atlas_int = np.round(atlas_data).astype(np.int32)
    return atlas_int, int(atlas_int.max())

def compute_centroids_for_labels(atlas_vox, labels):
    """Return centroid coordinates (voxel space) for each label in labels list/array"""
    coords = []
    for lab in labels:
        # center_of_mass returns (z,y,x) in voxel coordinates
        cm = center_of_mass(atlas_vox == lab)
        coords.append(cm)
    return np.array(coords)  # shape (n_labels, 3)

def run_pipeline(base_path, participant, participantDictionary,
                 dwi_fname, bval_fname, bvec_fname, atlas_fname,
                 out_dir=None,
                 fa_threshold=0.2,
                 seeds_density=2,  # seeds per voxel
                 step_size=0.5,
                 min_length=10,
                 downsample_resolutions=(256,128,64,32),
                 apply_smoothing=True,
                 smoothing_sigma=1.0,
                 log_normalize=True):
    """
    Full pipeline. All filepaths should be absolute or relative to base_path.
    """
    if out_dir is None:
        out_dir = os.path.join(base_path, 'output')
    os.makedirs(out_dir, exist_ok=True)

    dwi_path = os.path.join(base_path, 'dwi', dwi_fname)
    bval_path = os.path.join(base_path, 'dwi', bval_fname)
    bvec_path = os.path.join(base_path, 'dwi', bvec_fname)
    atlas_path = os.path.join(base_path, 'atlas', atlas_fname)

    print("Loading DWI and gradient data...")
    img = nib.load(dwi_path)
    data = img.get_fdata()
    affine = img.affine
    print(f" DWI shape: {data.shape}, affine:\n{affine}")

    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    # bvecs may be stored as columns; make sure shape is (N,3)
    if bvecs.shape[0] == 3 and bvecs.shape[1] == len(bvals):
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs)

    print("Loading atlas and resampling to DWI (if needed)...")
    atlas_img = nib.load(atlas_path)
    # Resample atlas to DWI voxel grid so label_volume and streamlines share voxel coordinates
    if not np.allclose(atlas_img.header.get_zooms()[:3], img.header.get_zooms()[:3]) \
       or atlas_img.shape != img.shape[:3]:
        print(" Resampling atlas -> DWI grid using nilearn.resample_to_img...")
        atlas_img = resample_to_img(atlas_img, img, interpolation='nearest')
        print(" Resampling done.")
    atlas_vox, maxlab = ensure_int_labels(atlas_img)

    print("Fitting tensor model and computing FA...")
    tenmodel = TensorModel(gtab)
    # fit on whole volume (or provide mask to speed up)
    tenfit = tenmodel.fit(data)
    fa = fractional_anisotropy(tenfit.evals)
    fa = np.nan_to_num(fa)

    # create stopping mask and seeds
    mask = fa > fa_threshold
    print(f"Seed density: {seeds_density} -> generating seeds from mask; mask voxels: {np.sum(mask)}")
    seeds = seeds_from_mask(mask, affine=affine, density=seeds_density)

    print("Computing peak directions (for deterministic tracking)...")
    # generate peaks structure used by LocalTracking
    peaks = peaks_from_model(model=tenmodel, data=data, sphere=default_sphere,
                             relative_peak_threshold=0.5, min_separation_angle=25,
                             mask=mask)

    print("Running LocalTracking (deterministic) ...")
    stopping_criterion = ThresholdStoppingCriterion(fa, fa_threshold)
    # LocalTracking yields streamlines in world coordinates when affine provided
    streamlines_trk = LocalTracking(peaks, stopping_criterion, seeds,
                                    affine=affine, step_size=step_size)
    streamlines = Streamlines(streamlines_trk)

    # Optionally filter very short streamlines
    streamlines = Streamlines([s for s in streamlines if len(s) >= min_length])
    print(f" Streamlines after length filter: {len(streamlines)}")

    if len(streamlines) == 0:
        raise RuntimeError("No streamlines generated. Check FA threshold, seeds density and data quality.")

    print("Computing connectivity matrix (label-volume based)...")
    # connectivity_matrix expects streamlines in world coordinates and a label_volume with same affine
    conn_full, mapping = connectivity_matrix(streamlines, affine=affine, label_volume=atlas_vox,
                                             return_mapping=True)

    # connectivity_matrix returns matrix with index up to max label (0..maxlab)
    # We'll extract rows/cols for actual labels present (>0)
    labels_present = np.array(sorted(np.unique(atlas_vox)))
    labels_present = labels_present[labels_present > 0]  # exclude background 0
    print(f" Number of atlas labels present (>0): {len(labels_present)}")

    # extract the submatrix for labels_present
    conn_sub = conn_full[np.ix_(labels_present, labels_present)].astype(np.float64)

    # symmetry enforcement
    conn_sub = (conn_sub + conn_sub.T) / 2.0

    # smoothing (optional)
    if apply_smoothing:
        print(f"Applying gaussian smoothing sigma={smoothing_sigma} to connectivity matrix")
        conn_sub_sm = gaussian_filter(conn_sub, sigma=smoothing_sigma)
    else:
        conn_sub_sm = conn_sub

    # robust normalization
    print("Normalizing connectivity matrix (log1p + percentile-scale)...")
    conn_norm = robust_normalize(conn_sub_sm, percentile=99.5, do_log=log_normalize)

    # Save original-resolution (label-based) connectome with mapping to atlas labels
    save_base = os.path.join(out_dir, f'connectome_{participantDictionary[participant]}_labels.npy')
    np.save(save_base, conn_norm.astype(np.float32))
    # Save also the labels mapping
    np.save(os.path.join(out_dir, f'connectome_{participantDictionary[participant]}_labels_index.npy'), labels_present.astype(np.int32))
    print(f"Saved label-based connectome ({conn_norm.shape}) and label index mapping.")

    # Downsample to target resolutions by clustering centroid coordinates of the labels
    print("Computing centroids for label clustering...")
    centroids = compute_centroids_for_labels(atlas_vox, labels_present)
    # centroids are (z,y,x) floats; convert to e.g. (x,y,z) or keep as is: relative distances preserved

    for res in downsample_resolutions:
        print(f"Downsampling to {res}x{res} via KMeans on region centroids...")
        # If number of labels less than res, just upsample via interpolation later - but typically SCHAEFER300 > res
        n_labels = len(labels_present)
        if n_labels <= res:
            print(f"  n_labels ({n_labels}) <= target res ({res}), skipping clustering, resampling matrix instead.")
            # use zoom to rescale conn_norm to (res,res)
            down = zoom(conn_norm, (res / conn_norm.shape[0], res / conn_norm.shape[1]), order=1)
            down = (down + down.T) / 2.0
            down = robust_normalize(down, percentile=99.5, do_log=False)  # already normalized
            np.save(os.path.join(out_dir, f'connectome_{participantDictionary[participant]}_{res}.npy'), down.astype(np.float32))
            continue

        # KMeans on centroids (use the 3D coords)
        kmeans = KMeans(n_clusters=res, random_state=42, n_init=10).fit(centroids)
        cluster_labels = kmeans.labels_  # length = n_labels

        # build aggregated matrix
        downsampled = np.zeros((res, res), dtype=np.float64)
        counts = np.zeros((res, res), dtype=np.int32)
        # map original region indices (0..n_labels-1) to cluster id
        for i_cluster in range(res):
            idx_i = np.where(cluster_labels == i_cluster)[0]  # positions in labels_present
            if idx_i.size == 0:
                continue
            for j_cluster in range(res):
                idx_j = np.where(cluster_labels == j_cluster)[0]
                if idx_j.size == 0:
                    continue
                # average across submatrix entries
                sub = conn_norm[np.ix_(idx_i, idx_j)]
                downsampled[i_cluster, j_cluster] = np.nanmean(sub)
                counts[i_cluster, j_cluster] = sub.size

        # ensure symmetry and normalize
        downsampled = (downsampled + downsampled.T) / 2.0
        downsampled = robust_normalize(downsampled, percentile=99.5, do_log=False)
        np.save(os.path.join(out_dir, f'connectome_{participantDictionary[participant]}_{res}.npy'), downsampled.astype(np.float32))
        print(f" Saved downsampled {res}x{res}")

    # create one upsampled matrix (example 512x512) by interpolation
    target_up = 512
    print(f"Upsampling (interpolating) original label-based matrix to {target_up}x{target_up} ...")
    upsampled = zoom(conn_norm, (target_up / conn_norm.shape[0], target_up / conn_norm.shape[1]), order=1)
    upsampled = (upsampled + upsampled.T) / 2.0
    upsampled = robust_normalize(upsampled, percentile=99.5, do_log=False)
    np.save(os.path.join(out_dir, f'connectome_{participantDictionary[participant]}_{target_up}.npy'), upsampled.astype(np.float32))
    print("Saved upsampled matrix.")

    print("All connectome matrices saved to:", out_dir)
    return {
        'conn_label_matrix': conn_norm,
        'labels_present': labels_present,
        'downsampled_resolutions': downsample_resolutions
    }


if __name__ == "__main__":

    participantDictionary = {
        'SNIPKPB8403': 'beRNN_01',
        'SNIPYL4AS03': 'beRNN_02',
        'SNIP6IECX03': 'beRNN_03',
        'SNIPDKHPB03': 'beRNN_04',
        'SNIP96WID03': 'beRNN_05'
    }

    # === Edit these filenames to match your directory layout ===
    participant = 'SNIP6IECX03'  # sample id - add manually
    base = fr"C:\Users\oliver.frank\Desktop\PyProjects\beRNN_bio\weightMatrices_dwi\{participantDictionary[participant]}\ses-03"
    # filenames (update to your exact names)
    dwi_file = f"025_sub-{participant}_diff_PA_257.nii.gz"
    bval_file = f"025_sub-{participant}_diff_PA_257.bval"
    bvec_file = f"025_sub-{participant}_diff_PA_257.bvec"
    atlas_file = "atlas_resampled_to_dwi.nii.gz"  # will be resampled if not matching

    out = run_pipeline(base, participant, participantDictionary,
                       dwi_fname=dwi_file,
                       bval_fname=bval_file,
                       bvec_fname=bvec_file,
                       atlas_fname=atlas_file,
                       out_dir=os.path.join(base, 'output'),
                       fa_threshold=0.20,
                       seeds_density=2,
                       step_size=0.5,
                       min_length=10,
                       downsample_resolutions=(256,128,64,32),
                       apply_smoothing=True,
                       smoothing_sigma=1.0,
                       log_normalize=True)
    print("Done.")



########################################################################################################################
# Visualize beRNN dti connectomes - scaled & binary version
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import os

def random_orthogonal(n, rng=None):
    """
    Draw a random orthogonal matrix distributed uniformly (Haar) over O(n).
    """
    if rng is None:
        rng = np.random.default_rng()          # or np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(H)
    # Make Q uniformly distributed by flipping signs so diag(R) > 0
    Q *= np.sign(np.diag(R))
    return Q


maskSize = 256
structuralMask = np.load(os.path.join(r'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_bio\weightMatrices_dwi\beRNN_05\ses-03\output', 'connectome_beRNN_05_256.npy'))

w_rec0_ = np.nan_to_num((structuralMask + structuralMask.T) / 2, nan=0.0, posinf=0.0, neginf=0.0)
# Draw a new random rotation each run (or per epoch if you like)
Q = random_orthogonal(w_rec0_.shape[0])
w_rec0 = Q @ w_rec0_ @ Q.T

# Normalize to controlled spectral radius (~1)
eigvals = np.linalg.eigvals(w_rec0)
max_eig = np.max(np.abs(eigvals))
if max_eig > 0:
    w_rec0 = w_rec0 / max_eig * 0.5

np.mean(w_rec0 != 0)

structuralMask_binary = w_rec0.copy()
counter1 = 0

for i in range(0, maskSize):
    for j in range(0, maskSize):
        if structuralMask[i, j] > 0.025:
            structuralMask_binary[i, j] = 1
            counter1 += 1
        else:
            structuralMask_binary[i, j] = 0

plt.figure(figsize=(8, 8))
plt.imshow(w_rec0, aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.title("Visualization of a structuralMask")
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(structuralMask_binary, aspect='auto', cmap='coolwarm')
plt.colorbar()
plt.title("Visualization of a structuralMask")
plt.show()


