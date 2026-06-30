"""Execution time analysis"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .core import save_plot, get_figure

def plot_time_analysis(results, save_dir: str):
    """Build time plots"""

    fig, axes = get_figure(1, 2, (12, 5))

    # 1. Time by agent (always useful)
    agent_build, agent_test = defaultdict(list), defaultdict(list)
    for r in results:
        agent_build[r.agent_id].append(r.build_time)
        agent_test[r.agent_id].append(r.test_time)

    agents = sorted(agent_build.keys())
    x = np.arange(len(agents))
    width = 0.35

    axes[0].bar(x - width/2, [np.mean(agent_build[a]) for a in agents],
                width, label='Build', color='skyblue')
    axes[0].bar(x + width/2, [np.mean(agent_test[a]) for a in agents],
                width, label='Test', color='lightgreen')
    axes[0].set_title('Average time by agent', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Agent {a}' for a in agents], rotation=45)
    axes[0].set_ylabel('Time (s)')
    axes[0].legend()

    # 2. Check if second plot makes sense
    train_sizes = [r.train_data_size for r in results]

    # If all sizes are the same - build boxplot instead of scatter
    if len(set(train_sizes)) == 1:
        # Boxplot of time distribution
        build_times = [r.build_time for r in results]

        axes[1].boxplot(build_times)
        axes[1].set_title('Build time distribution', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Time (s)')
        axes[1].set_xticklabels(['Build'])

        # Add statistics
        stats = f'μ={np.mean(build_times):.2f}s\nσ={np.std(build_times):.2f}s'
        axes[1].text(0.95, 0.95, stats, transform=axes[1].transAxes,
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Scatter plot with trend (as before)
        build_times = [r.build_time for r in results]
        comp_values = [r.representatives_count / r.train_data_size for r in results]

        scatter = axes[1].scatter(train_sizes, build_times, c=comp_values,
                                  cmap='coolwarm', alpha=0.6)
        axes[1].set_title('Build time vs Data size', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Training set size')
        axes[1].set_ylabel('Build time (s)')
        plt.colorbar(scatter, ax=axes[1], label='Compression')

        # Trend line
        if len(train_sizes) > 1:
            z = np.polyfit(train_sizes, build_times, 1)
            x_trend = np.linspace(min(train_sizes), max(train_sizes), 100)
            axes[1].plot(x_trend, np.poly1d(z)(x_trend), "r--", alpha=0.8,
                        label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')
            axes[1].legend()

    save_plot(fig, save_dir, 'time_analysis.png')
