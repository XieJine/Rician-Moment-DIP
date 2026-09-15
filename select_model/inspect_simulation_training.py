"""Inspect simulation metrics and report representative checkpoints.

User configuration:
    Key parameters to edit: metric/checkpoint paths, save directory, checkpoint interval, iteration range, b-value metadata, mask options, plot limits, and selection window.
    Relative paths assume execution from the repository root.
"""

import numpy as np
import matplotlib.pyplot as plt

# Load metric arrays.
path_main ="outputs/DIP_M1_W/generate_mean_dwi_input2/psnr_rmse/level_5/"
a1 = np.load(path_main+"psnr_level_5.npy")
a = np.load(path_main+"loss_level_5.npy")

# Sample the saved metrics at the configured interval.
x = [i for i in range(0, a.shape[0], 40)]
b = a[0:a.shape[0]:40]  # loss values
b1 = a1[0:a1.shape[0]:40]  # PSNR values

# Read the final saved iteration.
last_iteration = x[-1]  # Final sampled iteration.
last_loss = b[-1]
last_psnr = b1[-1]

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
print(f"Iteration range: 0 - {x[-1]} (sampled)")
print(f"Final iteration: {last_iteration}")
print(f"Final-iteration PSNR: {last_psnr:.4f}")
print(f"Final-iteration loss: {last_loss:.6f}")
print(f"Minimum loss iteration: {min_iteration}")
print(f"Minimum loss: {min_loss:.6f}")
print(f"PSNR at minimum loss: {psnr_at_min_loss:.4f}")
print(f"Maximum PSNR iteration: {max_iteration}")
print(f"Maximum PSNR: {max_psnr:.4f}")

# Query PSNR and loss at a selected iteration.
def query_psnr_at_iteration(iteration):
    """Return PSNR and loss at a selected iteration."""
    if iteration in x:
        idx = x.index(iteration)
        psnr_value = b1[idx]
        loss_value = b[idx]
        print(f"\nQuery result - iteration {iteration}:")
        print(f"  PSNR: {psnr_value:.4f}")
        print(f"  Loss: {loss_value:.6f}")
        return psnr_value, loss_value
    else:
        print(f"Error: iteration {iteration} is not present in the sampled data")
        print(f"Available iteration range: {x[0]} - {x[-1]}, step: 40")
        return None, None

# Query representative checkpoints.
print("\n" + "="*50)
print("Representative checkpoint results:")
print("="*50)
key_iterations = [min_iteration, max_iteration, last_iteration, 20000, 30000, 50000]
queried_points = []

for iter_num in key_iterations:
    if iter_num in x:
        psnr_val, loss_val = query_psnr_at_iteration(iter_num)
        if psnr_val is not None:
            queried_points.append((iter_num, psnr_val, loss_val))

# Optional interactive query.
print("\n" + "="*50)
print("Interactive query (enter -1 to exit):")
print("="*50)

while True:
    try:
        user_input = input("Enter an iteration: ")
        if user_input == '-1':
            break
        iteration = int(user_input)
        query_psnr_at_iteration(iteration)
    except ValueError:
        print("Enter a valid integer.")
    except KeyboardInterrupt:
        break

# Plot training curves.
plt.figure(figsize=(16, 6))

# Plot the loss curve.
plt.subplot(1, 2, 1)
plt.plot(x, b, 'b-', linewidth=1.5, label='Loss')
# Mark the minimum-loss point.
plt.plot(min_iteration, min_loss, 'yo', markersize=8, label='Min Loss')
plt.text(min_iteration, min_loss + 0.02, 
         f'Min Loss\nIter: {min_iteration}\nPSNR: {psnr_at_min_loss:.2f}', 
         ha='center', va='bottom', fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
# Mark the final iteration.
plt.plot(last_iteration, last_loss, 'ks', markersize=8, label='Final Iteration')
plt.text(last_iteration, last_loss + 0.02, 
         f'Final\nIter: {last_iteration}\nPSNR: {last_psnr:.2f}', 
         ha='center', va='bottom', fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.ylim(0,1)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot the PSNR curve.
plt.subplot(1, 2, 2)
plt.plot(x, b1, 'r-', linewidth=1.5, label='PSNR')
# Mark PSNR at the minimum-loss point.
plt.plot(min_iteration, psnr_at_min_loss, 'yo', markersize=8, label='PSNR at Min Loss')
plt.text(min_iteration, psnr_at_min_loss - 0.5, 
         f'Min Loss\nIter: {min_iteration}\nPSNR: {psnr_at_min_loss:.2f}', 
         ha='center', va='top', fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
# Mark the maximum PSNR.
plt.plot(max_iteration, max_psnr, 'go', markersize=8, label='Max PSNR')
plt.text(max_iteration, max_psnr + 0.3, 
         f'Max PSNR\nIter: {max_iteration}\nPSNR: {max_psnr:.2f}', 
         ha='center', va='bottom', fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
# Mark the final iteration.
plt.plot(last_iteration, last_psnr, 'ks', markersize=8, label='Final Iteration')
plt.text(last_iteration, last_psnr - 0.5, 
         f'Final\nIter: {last_iteration}\nPSNR: {last_psnr:.2f}', 
         ha='center', va='top', fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.ylim(30, 42)
plt.xlabel('Iteration')
plt.ylabel('PSNR (dB)')
plt.title('PSNR vs Iteration')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(path_main+'combined_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed report.
print("\n" + "="*70)
print("Detailed performance report:")
print("="*70)
print(f"{'Iteration':<10} {'PSNR (dB)':<12} {'Loss':<12} {'Note':<20}")
print("-" * 70)

report_data = [
    (min_iteration, psnr_at_min_loss, min_loss, "minimum-loss point"),
    (max_iteration, max_psnr, b[max_idx], "maximum-PSNR point"),
    (last_iteration, last_psnr, last_loss, "Final iteration")
]

# Add other queried checkpoints.
for iter_num, psnr_val, loss_val in queried_points:
    if iter_num not in [min_iteration, max_iteration, last_iteration]:
        remark = "queried point"
        if iter_num == 20000:
            remark = "early training"
        elif iter_num == 30000:
            remark = "middle training"
        elif iter_num == 50000:
            remark = "late training"
        report_data.append((iter_num, psnr_val, loss_val, remark))

# Sort by iteration.
report_data.sort()

for iter_num, psnr_val, loss_val, remark in report_data:
    print(f"{iter_num:<10} {psnr_val:<12.4f} {loss_val:<12.6f} {remark:<20}")

print("="*70)

# Performance comparison.
print("\nPerformance comparison.:")
print(f"Final PSNR vs maximum PSNR: {last_psnr:.4f} dB / {max_psnr:.4f} dB (difference: {max_psnr - last_psnr:.4f} dB)")
print(f"Final loss vs minimum loss: {last_loss:.6f} / {min_loss:.6f} (difference: {last_loss - min_loss:.6f})")

# Save the report.
with open(path_main+'performance_report.txt', 'w') as f:
    f.write("PSNR and loss performance report\n")
    f.write("=" * 60 + "\n")
    f.write(f"Iteration range: 0 - {x[-1]} (sampled)\n\n")
    f.write(f"{'Iteration':<10} {'PSNR (dB)':<12} {'Loss':<12} {'Note':<20}\n")
    f.write("-" * 60 + "\n")
    for iter_num, psnr_val, loss_val, remark in report_data:
        f.write(f"{iter_num:<10} {psnr_val:<12.4f} {loss_val:<12.6f} {remark:<20}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Performance comparison.:\n")
    f.write(f"Final PSNR vs maximum PSNR: {last_psnr:.4f} dB / {max_psnr:.4f} dB (difference: {max_psnr - last_psnr:.4f} dB)\n")
    f.write(f"Final loss vs minimum loss: {last_loss:.6f} / {min_loss:.6f} (difference: {last_loss - min_loss:.6f})\n")

print(f"\nReport saved to: "+path_main+"performance_report.txt")
