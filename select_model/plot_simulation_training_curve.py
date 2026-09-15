"""Plot simulation loss and PSNR training curves.

User configuration:
    Key parameters to edit: metric/checkpoint paths, save directory, checkpoint interval, iteration range, b-value metadata, mask options, plot limits, and selection window.
    Relative paths assume execution from the repository root.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load metric arrays.
a1 = np.load("outputs/DIP_M1_W/generate_data_2/noisemap/psnr_rmse/level_5/psnr_level_5.npy")
a = np.load("outputs/DIP_M1_W/generate_data_2/noisemap/psnr_rmse/level_5/loss_level_5.npy")

# Sample the saved metrics at the configured interval.
x = [i for i in range(0, a.shape[0], 40)]
b = a[0:a.shape[0]:40]  # loss values
b1 = a1[0:a1.shape[0]:40]  # PSNR values

# Locate the minimum loss.
min_idx = np.argmin(b)
min_iteration = x[min_idx]
min_loss = np.min(b)

# Read PSNR at the minimum-loss iteration.
psnr_at_min_loss = b1[min_idx]

# Locate the maximum PSNR.
max_idx = np.argmax(b1)
max_iteration = x[max_idx]
max_psnr = np.max(b1)

print("Key metrics:")
print(f"Minimum loss iteration: {min_iteration}")
print(f"Minimum loss: {min_loss:.6f}")
print(f"PSNR at minimum loss: {psnr_at_min_loss:.4f}")
print(f"Maximum PSNR iteration: {max_iteration}")
print(f"Maximum PSNR: {max_psnr:.4f}")

# Plot training curves.
plt.figure(figsize=(14, 6))

# Plot the loss curve.
plt.subplot(1, 2, 1)
plt.plot(x, b, 'b-', linewidth=1.5, label='Loss')
plt.plot(min_iteration, min_loss, 'yo', markersize=8)
plt.text(min_iteration, min_loss + 0.02, 
         f'Iter: {min_iteration}\nPSNR: {psnr_at_min_loss:.2f}\nLoss: {min_loss:.4f}', 
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.ylim(1, 1.5)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.grid(True, alpha=0.3)

# Plot the PSNR curve.
plt.subplot(1, 2, 2)
plt.plot(x, b1, 'r-', linewidth=1.5, label='PSNR')
# Mark PSNR at the minimum-loss point.
plt.plot(min_iteration, psnr_at_min_loss, 'yo', markersize=8)
plt.text(min_iteration, psnr_at_min_loss - 0.5, 
         f'Iter: {min_iteration}\nPSNR: {psnr_at_min_loss:.2f}\nLoss: {min_loss:.4f}', 
         ha='center', va='top', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
# Mark the maximum PSNR.
plt.plot(max_iteration, max_psnr, 'go', markersize=8)
plt.text(max_iteration, max_psnr + 0.3, 
         f'Iter: {max_iteration}\nMax PSNR: {max_psnr:.2f}', 
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
plt.ylim(35, 40)
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Iteration')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/DIP_M1_W/generate_data_2/noisemap/psnr_rmse/level_5/combined_plots.png', dpi=300, bbox_inches='tight')
plt.show()
