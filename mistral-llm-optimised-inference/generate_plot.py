"""
generate_plot.py : Generates inference throughput benchmark visualization.
Run from inside the mistral-llm-optimised-inference/ directory.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid")

categories = ['Baseline\n(FP32, No Batching)', '4-bit Quant\n(Batch=8)', '4-bit Quant\n(Batch=32)']
throughput = [28, 112, 215]
colors = ["#e74c3c", "#f39c12", "#27ae60"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(categories, throughput, color=colors, width=0.5, zorder=3)

ax.axhline(y=200, color='black', linestyle='--', linewidth=1.5, zorder=4, label='Target: 200 tok/s')
ax.set_title('Mistral-7B Inference Throughput on NVIDIA T4', fontsize=15, fontweight='bold', pad=16)
ax.set_ylabel('Tokens / Second', fontsize=12)
ax.set_ylim(0, 260)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

for bar, val in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
            f'{val} tok/s', ha='center', va='bottom', fontsize=12, fontweight='bold')

target_patch = mpatches.Patch(color='black', linestyle='--', label='Target: 200 tok/s')
ax.legend(handles=[target_patch], fontsize=11)

plt.tight_layout()

# Save directly into the mistral project folder
out_path = os.path.join(os.path.dirname(__file__), 'throughput_benchmark.png')
plt.savefig(out_path, dpi=300)
print(f"Plot saved to: {out_path}")
