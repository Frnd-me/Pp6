import json
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict


def plot_benchmarks():
    # results[size][algo] = {'x': threads, 'y': time}
    results = defaultdict(lambda: defaultdict(lambda: {'x': [], 'y': []}))

    files = glob.glob("threads_*.json")
    # Sort files by thread count extracted from filename
    files.sort(key=lambda x: int(re.search(r'threads_(\d+)', x).group(1)))

    if not files:
        print("No benchmark files found.")
        return

    # Parse all data points
    all_sizes = set()
    for file_path in files:
        threads = int(re.search(r'threads_(\d+)', file_path).group(1))
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                for bench in data.get('benchmarks', []):
                    # Regex to capture "Algorithm" and "Size" from "BM_Algo/Size/..."
                    match = re.match(r'BM_([^/]+)/(\d+)', bench['name'])
                    if match:
                        algo = match.group(1)
                        size = int(match.group(2))
                        all_sizes.add(size)

                        time_ms = bench['real_time'] / 1e6
                        results[size][algo]['x'].append(threads)
                        results[size][algo]['y'].append(time_ms)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {file_path}: {e}")

    sorted_sizes = sorted(list(all_sizes))
    num_sizes = len(sorted_sizes)

    # Calculate grid dimensions for subplots
    cols = 2
    rows = (num_sizes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), sharex=True)
    axes = axes.flatten()

    for i, size in enumerate(sorted_sizes):
        ax = axes[i]
        size_data = results[size]

        for algo, values in size_data.items():
            ax.plot(values['x'], values['y'], marker='o', label=algo)

        ax.set_title(f"Problem Size: {size}")
        ax.set_yscale("log")
        ax.set_ylabel("Time (ms)")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize='small')

    # Global X label
    for ax in axes[-cols:]:
        ax.set_xlabel("Threads")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_benchmarks()
