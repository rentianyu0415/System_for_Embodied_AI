import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Llama-7B", "fingpt-LoRA", "FlagAlpha-LoRA", "Q8_0", "Q4_K_M", "Q2_K"]
memory_usage = [5942, 5936, 6031, 5999, 6074, 4622]
generation_speed = [3.4, 3.5, 3.6, 12.4, 53.5, 65.6]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
bar_width = 0.4

# Memory usage bars
bars1 = ax1.bar(x - bar_width / 2, memory_usage, bar_width, color="#4C72B0", label="Memory Usage /MiB")
ax1.set_ylabel("Memory Usage /MiB", color="#4C72B0", fontsize=12)
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_ylim(0, max(memory_usage) * 1.15)

# Add value labels on memory bars
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
             f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9, color="#4C72B0")

# Generation speed bars on secondary y-axis
ax2 = ax1.twinx()
bars2 = ax2.bar(x + bar_width / 2, generation_speed, bar_width, color="#DD8452", label="Generation t/s")
ax2.set_ylabel("Generation Speed (t/s)", color="#DD8452", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#DD8452")
ax2.set_ylim(0, max(generation_speed) * 1.2)

# Add value labels on speed bars
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, color="#DD8452")

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_xlabel("Model", fontsize=12)

fig.suptitle("Model Comparison: Memory Usage vs Generation Speed", fontsize=14, fontweight="bold")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

plt.tight_layout()
import os
save_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
plt.show()
print("Saved to model_comparison.png")
