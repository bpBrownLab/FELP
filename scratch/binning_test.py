#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#upload data skipping the first row
data = np.loadtxt("/home/murdocte/Projects/Rotation/GLY/output0508/stride_2/omm_energies.csv", delimiter=",", skiprows=1)
BINS=180

#create a 2d array of phi and psi values    
stride = data[:, 0]
og_energy = data[:, 1]
mut_energy = data[:, 2]
energy_diff1 = data[:, 3]
phi = data[:, 4]
psi = data[:, 5]
p_prob = []
kt = 2.479 # should correspond to KJ/mol; 0.592 for kcal/mol
epsilon = 1e-6 # Epsilon for numerical stability
delta_e_sum = 0

assert len(phi) == len(psi)
assert len(phi) == len(mut_energy)
assert len(phi) == len(og_energy)
assert len(phi) == len(energy_diff1)
assert len(phi) == len(stride)

#save x and y edges, counts, and densities from histogram
counts, xedges, yedges = np.histogram2d(phi, psi, bins=BINS)

#try histogram with densities
density, xedges, yedges = np.histogram2d(phi, psi, bins=BINS, density=True)

# print shape of counts and edges
print(f"counts.shape: {counts.shape}")
print(f"xedges.shape: {xedges.shape}")
print(f"yedges.shape: {yedges.shape}")

#calculate the probabilities for each bin 
sum_counts = np.sum(counts)
print(f"sum_counts: {sum_counts}")
bin_probabilities = np.array(counts / sum_counts)
print(f"bin_probabilities.shape: {bin_probabilities.shape}")
print(f"bin_probabilities: {bin_probabilities}")
print(f"bin_probabilities.sum(): {bin_probabilities.sum()}")

# Show the indices of the bins with below epsilon probability
below_epsilon_indices = np.where(bin_probabilities < epsilon)
print(f"below_epsilon_indices: {below_epsilon_indices}")

# create a 2D numpy array to track counts in each histogram bin from the input data
# counts_array = np.zeros((len(xedges)-1, len(yedges)-1))
energy_diff_sums = np.zeros((len(xedges)-1, len(yedges)-1))

# make energy diffs relative to the minimum energy diff
min_energy_diff = np.min(energy_diff1)
energy_diff1 = energy_diff1 - min_energy_diff

# for i in range(len(phi)):
#     for j in range(len(xedges)-1):
#         for k in range(len(yedges)-1):
#             if phi[i] >= xedges[j] and phi[i] <= xedges[j+1] and psi[i] >= yedges[k] and psi[i] <= yedges[k+1]:
#                 counts_array[j][k] += 1
#                 energy_diff_sums[j][k] += energy_diff1[i]

# Vectorized approach
# Subtract 1 because digitize returns 1-based indexing and we need 0-based for array access
# Also, np.digitize needs the bins to be monotonically increasing or decreasing. xedges and yedges are already sorted.
x_indices = np.digitize(phi, xedges[:-1]) - 1
y_indices = np.digitize(psi, yedges[:-1]) - 1

# Ensure indices are within bounds (0 to len(bins)-1)
# np.digitize handles values outside the bins by assigning them to 0 or len(bins).
# We need to clip them to the valid range of indices.
# xedges has shape (nbins+1,), so valid indices for counts_array/energy_diff_sums are 0 to nbins-1
x_indices = np.clip(x_indices, 0, len(xedges) - 2) # -2 because xedges has nbins+1 elements, so indices are 0 to nbins-1
y_indices = np.clip(y_indices, 0, len(yedges) - 2) # -2 because yedges has nbins+1 elements, so indices are 0 to nbins-1


# Increment counts for each bin
# np.add.at performs unbuffered in-place addition, which is what we need here.
np.add.at(counts, (x_indices, y_indices), 1)

# Sum energy differences for each bin
np.add.at(energy_diff_sums, (x_indices, y_indices), energy_diff1)


#calculate the probabilities for each bin
# Handle potential division by zero if some bins are empty
energy_diff_averages = np.zeros_like(counts)
non_empty_bins = counts > 0
energy_diff_averages[non_empty_bins] = energy_diff_sums[non_empty_bins] / counts[non_empty_bins]
print(f"energy_diff_averages.shape: {energy_diff_averages.shape}")
print(f"energy_diff_averages: {energy_diff_averages}")
print(f"energy_diff_averages.sum(): {energy_diff_averages.sum()}")

# Iterate over each bin and calculate the reweighted probabilities
reweighted_probabilities = np.zeros((len(xedges)-1, len(yedges)-1))
for i in range(len(xedges)-1):
    for j in range(len(yedges)-1):
        bin_prob = bin_probabilities[i][j]
        bin_energy_diff = energy_diff_averages[i][j]
        pstar_value = bin_prob * np.exp(-kt * bin_energy_diff)
        reweighted_probabilities[i][j] = pstar_value

# Normalize the reweighted probabilities
sum_reweighted_prob = np.sum(reweighted_probabilities)
reweighted_probabilities = reweighted_probabilities / sum_reweighted_prob
print(f"reweighted_probabilities.shape: {reweighted_probabilities.shape}")
print(f"reweighted_probabilities: {reweighted_probabilities}")
print(f"reweighted_probabilities.sum(): {reweighted_probabilities.sum()}")

# Make a 2D heatmap of the original probabilities
original_free_energy_surface = -kt * np.log(bin_probabilities + epsilon)
plt.imshow(original_free_energy_surface.T, cmap="viridis")

# Set colorbar limits
plt.clim(0, 30)
plt.colorbar()
plt.xlabel("phi")
plt.ylabel("psi")
plt.title("Glycine free energy surface")
plt.savefig("glycine_free_energy_surface.png", dpi=300)
plt.close()

# Make a 2D heatmap of the free energy surface
free_energy_surface = -kt * np.log(reweighted_probabilities + epsilon)

# Set max value to 10 if it's greater than 10
free_energy_surface = np.clip(free_energy_surface, 0, 30)

print(f"free_energy_surface.shape: {free_energy_surface.shape}")
print(f"free_energy_surface: {free_energy_surface}")
print(f"free_energy_surface.min(): {free_energy_surface.min()}")
print(f"free_energy_surface.max(): {free_energy_surface.max()}")
plt.imshow(free_energy_surface.T, cmap="viridis")

# Set colorbar limits
plt.clim(0, 30)
plt.colorbar()
plt.xlabel("phi")
plt.ylabel("psi")
plt.title("Alanine free energy surface from Glycine")
plt.savefig("alanine_free_energy_surface.png", dpi=300)
plt.close()