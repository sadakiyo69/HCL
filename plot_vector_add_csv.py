import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Eingabe: CSV-Datei mit einer Zeile pro Messung, z. B. "results.csv":
#
#   elements,blocksize,gpu_ms,cpu_ms
#   1000,128,0.04,0.01
#   1000,256,0.04,0.01
#   1000000,128,0.60,9.50
#   1000000,256,0.55,9.50
#   ...
#
# Zeiten in ms. Die CPU-Zeit hängt nicht von der Blocksize ab,
# sie darf in jeder Zeile stehen (wird pro Elementanzahl gemittelt).
# ------------------------------------------------------------
CSV_FILE = "results.csv"

df = pd.read_csv(CSV_FILE)

# ------------------------------------------------------------
# PLOT 1: Zeit über Anzahl Elemente (CPU vs. GPU je Blocksize)
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

cpu = df.groupby("elements")["cpu_ms"].mean()
plt.plot(cpu.index, cpu.values, marker="o", color="black", linewidth=2, label="CPU")

for bs, group in df.groupby("blocksize"):
    group = group.sort_values("elements")
    plt.plot(group["elements"], group["gpu_ms"], marker="s", label=f"GPU (Blocksize {bs})")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Amount of elements")
plt.ylabel("Execution time [ms]")
plt.title("Vector Addition: CPU vs. GPU")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plot_cpu_vs_gpu.png", dpi=150)

# ------------------------------------------------------------
# PLOT 2: GPU-Zeit über Blocksize (eine Linie pro Elementanzahl)
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

for n, group in df.groupby("elements"):
    group = group.sort_values("blocksize")
    plt.plot(group["blocksize"], group["gpu_ms"], marker="o", label=f"N = {n:,}")

block_sizes = sorted(df["blocksize"].unique())
plt.xscale("log", base=2)
plt.yscale("log")
plt.xticks(block_sizes, [str(b) for b in block_sizes])
plt.xlabel("Blocksize (Threads per Block)")
plt.ylabel("GPU Execution time [ms]")
plt.title("Vector Addition: Influence of blocksize")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plot_blocksize.png", dpi=150)

plt.show()