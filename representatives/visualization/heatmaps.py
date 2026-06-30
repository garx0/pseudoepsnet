"""Heatmaps for accuracy, coverage, compression"""

import numpy as np
import matplotlib.pyplot as plt
from .core import save_plot, get_figure

def get_heatmap_data(results, agents, iterations, metric_func):
    """Create matrix for heatmap"""
    matrix = np.zeros((len(agents), len(iterations)))
    for r in results:
        i = agents.index(r.agent_id)
        j = iterations.index(r.iteration_id)
        matrix[i, j] = metric_func(r)
    return matrix


def plot_heatmaps(results, save_dir, agents, iterations):
    """Heatmaps for accuracy, coverage, compression"""

    # Data for heatmaps
    data = [
        ('Accuracy', 'RdYlGn', lambda r: r.metrics.accuracy),
        ('Coverage', 'Blues', lambda r: r.metrics.coverage),
        ('Compression', 'YlOrRd', lambda r: r.representatives_count / r.train_data_size)
    ]

    fig, axes = get_figure(1, 3, (18, 6))

    for ax, (title, cmap, func) in zip(axes, data):
        matrix = get_heatmap_data(results, agents, iterations, func)
        im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        ax.set(title=title, xlabel='Iterations', ylabel='Agents')
        ax.set_xticks(range(len(iterations)))
        ax.set_xticklabels(iterations)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        plt.colorbar(im, ax=ax)

        # Add values
        for i in range(len(agents)):
            for j in range(len(iterations)):
                if matrix[i, j] > 0:
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                           color='black', fontsize=9 if len(agents)*len(iterations) < 50 else 7)

    save_plot(fig, save_dir, 'heatmaps.png')
