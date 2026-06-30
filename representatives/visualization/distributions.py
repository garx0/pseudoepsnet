"""Distribution plots"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .core import save_plot, get_figure

def plot_distributions(results, save_dir):
    """Metric distributions"""

    fig, axes = get_figure(2, 3, (15, 10))

    # Histogram data
    hist_data = [
        ([r.metrics.accuracy for r in results], 'Accuracy', 'skyblue', axes[0, 0]),
        ([r.metrics.coverage for r in results], 'Coverage', 'lightgreen', axes[0, 1]),
        ([r.representatives_count / r.train_data_size for r in results],
         'Compression ratio', 'salmon', axes[0, 2])
    ]

    for values, title, color, ax in hist_data:
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color=color)
        ax.set_title(f'Distribution of {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(values), color='red', linestyle='--',
                  label=f'Mean: {np.mean(values):.3f}')
        ax.legend()

    # Scatter plots
    acc_values = [r.metrics.accuracy for r in results]
    comp_values = [r.representatives_count / r.train_data_size for r in results]
    build_times = [r.build_time for r in results]
    n_reps = [r.representatives_count for r in results]

    scatter1 = axes[1, 0].scatter(n_reps, build_times, c=acc_values, cmap='viridis', alpha=0.6)
    axes[1, 0].set_title('Build time vs Representatives', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of representatives')
    axes[1, 0].set_ylabel('Build time (s)')
    plt.colorbar(scatter1, ax=axes[1, 0], label='Accuracy')

    scatter2 = axes[1, 1].scatter(comp_values, acc_values, c=build_times, cmap='plasma', alpha=0.6)
    axes[1, 1].set_title('Accuracy vs Compression ratio', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Compression Ratio')
    axes[1, 1].set_ylabel('Accuracy')
    plt.colorbar(scatter2, ax=axes[1, 1], label='Build time')

    # Boxplot by agent
    agent_data = defaultdict(list)
    for r in results:
        agent_data[r.agent_id].append(r.metrics.accuracy)

    agents_sorted = sorted(agent_data.keys())
    axes[1, 2].boxplot([agent_data[a] for a in agents_sorted],
                       labels=[f'Agent {a}' for a in agents_sorted])
    axes[1, 2].set_title('Accuracy by agent', fontsize=12, fontweight='bold')
    axes[1, 2].tick_params(axis='x', rotation=45)

    save_plot(fig, save_dir, 'distributions.png')
