#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#upload data skipping the first row
data = np.loadtxt("omm_energies.csv", delimiter=",", skiprows=1)
BINS=180
CLIP=15

#create a 2d array of phi and psi values    
stride = data[:, 0]
og_energy = data[:, 1]
mut_energy = data[:, 2]
energy_diff1 = data[:, 3]
phi = data[:, 4]
psi = data[:, 5]
p_prob = []
kt = 2.479 # should correspond to KJ/mol; 0.592 for kcal/mol
epsilon = 1e-5 # Epsilon for numerical stability
delta_e_sum = 0

assert len(phi) == len(psi)
assert len(phi) == len(mut_energy)
assert len(phi) == len(og_energy)
assert len(phi) == len(energy_diff1)
assert len(phi) == len(stride)

#save x and y edges, counts, and densities from histogram
counts, xedges, yedges = np.histogram2d(phi, psi, bins=BINS)

# print shape of counts and edges
print(f"counts.shape: {counts.shape}")
print(f"xedges.shape: {xedges.shape}")
print(f"yedges.shape: {yedges.shape}")

#calculate the probabilities for each bin for Glycine (P_gly_bin)
sum_counts = np.sum(counts)
print(f"sum_counts: {sum_counts}")
bin_probabilities = np.array(counts / sum_counts)
print(f"bin_probabilities.shape: {bin_probabilities.shape}")
print(f"bin_probabilities: {bin_probabilities}")
print(f"bin_probabilities.sum(): {bin_probabilities.sum()}")

# Show the indices of the bins with below epsilon probability
below_epsilon_indices = np.where(bin_probabilities < epsilon)
print(f"below_epsilon_indices: {below_epsilon_indices}")

# Determine 0-based bin indices for each frame's phi and psi value
x_indices = np.digitize(phi, xedges[:-1]) - 1
y_indices = np.digitize(psi, yedges[:-1]) - 1

# Clip indices to ensure they are within the valid range [0, BINS-1]
x_indices = np.clip(x_indices, 0, BINS - 1)
y_indices = np.clip(y_indices, 0, BINS - 1)

# Shift energy differences by their minimum for numerical stability.
min_energy_diff = np.min(energy_diff1)
shifted_energy_diff_per_frame = energy_diff1 - min_energy_diff

# Calculate the Boltzmann factor for each frame using the shifted energy difference
frame_boltzmann_factors = np.exp(-shifted_energy_diff_per_frame / kt)

# Sum these frame-wise Boltzmann factors within each bin
sum_boltzmann_factors_in_bin = np.zeros_like(counts, dtype=float)
np.add.at(sum_boltzmann_factors_in_bin, (x_indices, y_indices), frame_boltzmann_factors)

# Calculate W_b = <exp(-DeltaE_shifted / kt)>_bin for each bin
W_b_values = np.zeros_like(counts, dtype=float)
non_empty_bins_mask = counts > 0
W_b_values[non_empty_bins_mask] = sum_boltzmann_factors_in_bin[non_empty_bins_mask] / counts[non_empty_bins_mask]

# Calculate unnormalized reweighted probabilities for Alanine: P_ala_unnormalized = P_gly * W_b
reweighted_probabilities_unnormalized = bin_probabilities * W_b_values

# Normalize the reweighted probabilities
sum_total_reweighted_prob = np.sum(reweighted_probabilities_unnormalized)

if sum_total_reweighted_prob > epsilon:
    reweighted_probabilities = reweighted_probabilities_unnormalized / sum_total_reweighted_prob
else:
    print("Warning: Sum of unnormalized reweighted probabilities is close to zero. Resulting probabilities might be meaningless.")
    reweighted_probabilities = np.zeros_like(counts, dtype=float)

# Make a 2D heatmap of the original probabilities
# Initialize with NaNs
original_free_energy_surface = np.full_like(bin_probabilities, np.nan, dtype=float)

# Create a mask for probabilities strictly greater than epsilon
mask_original = bin_probabilities > epsilon

# Calculate free energy only for bins where probability > epsilon
original_free_energy_surface[mask_original] = -kt * np.log(bin_probabilities[mask_original])

# Normalize the original free energy surface to start at 0, ignoring NaNs
min_val_original = np.nanmin(original_free_energy_surface)
if np.isfinite(min_val_original):
    original_free_energy_surface -= min_val_original

# Make a 2D heatmap of the original probabilities
plt.imshow(
    original_free_energy_surface.T,
    cmap="jet",
    origin='lower',
    extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]]
)

# Set colorbar limits
plt.clim(0, CLIP)
plt.colorbar(label="Free Energy (kJ/mol)")
plt.xlabel("$\psi$ (degrees)")
plt.ylabel("$\phi$ (degrees)")
plt.title("Glycine Free Energy Surface")
plt.savefig("glycine_free_energy_surface.jpg", dpi=300)
plt.close()

# Make a 2D heatmap of the free energy surface
# Initialize with NaNs
free_energy_surface = np.full_like(reweighted_probabilities, np.nan, dtype=float)

# Create a mask for probabilities strictly greater than epsilon
mask_reweighted = reweighted_probabilities > epsilon

# Calculate free energy only for bins where probability > epsilon
free_energy_surface[mask_reweighted] = -kt * np.log(reweighted_probabilities[mask_reweighted])

# Normalize the reweighted free energy surface to start at 0, ignoring NaNs
min_val_reweighted = np.nanmin(free_energy_surface)
if np.isfinite(min_val_reweighted):
    free_energy_surface -= min_val_reweighted

print(f"free_energy_surface.shape: {free_energy_surface.shape}")
print(f"free_energy_surface (finite min): {np.nanmin(free_energy_surface)}")
print(f"free_energy_surface (finite max): {np.nanmax(free_energy_surface)}")
plt.imshow(
    free_energy_surface.T,
    cmap="jet",
    origin='lower',
    extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]]
)

# Set colorbar limits
plt.clim(0, CLIP)
plt.colorbar(label="Free Energy (kJ/mol)")
plt.xlabel("$\psi$ (degrees)")
plt.ylabel("$\phi$ (degrees)")
plt.title("Alanine Free Energy Surface from Glycine")
plt.savefig("alanine_free_energy_surface.jpg", dpi=300)
plt.close()
