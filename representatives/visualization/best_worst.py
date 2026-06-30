import numpy as np
import matplotlib.pyplot as plt
from .core import save_plot, get_figure


def plot_best_worst(results, save_dir):
    """
    Combined comparison of best and worst results:
    - Left: bar charts of metrics
    - Right: table with full data
    """

    # Select results
    sorted_res = sorted(results, key=lambda x: x.metrics.accuracy)
    best_worst = sorted_res[-3:][::-1] + sorted_res[:3]
    titles = ['Best', '2nd', '3rd', 'Worst', '2nd worst', '3rd worst']

    fig = plt.figure(figsize=(16, 10))

    # Create grid: 2x3 for plots + space for table
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Plots for each result (2x3)
    for idx, (res, title) in enumerate(zip(best_worst, titles)):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])

        m = res.metrics
        metrics = ['Acc', 'Cov', 'Comp']
        values = [m.accuracy, m.coverage, m.compression_ratio]
        colors = ['#2ecc71' if v > 0.7 else '#f39c12' if v > 0.4 else '#e74c3c' for v in values]

        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Value')
        ax.set_title(f'{title}\nA{res.agent_id} I{res.iteration_id}', fontsize=10)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., v + 0.02,
                   f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # Table with full data at the bottom
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    table_data = []
    for res, title in zip(best_worst, titles):
        m = res.metrics
        table_data.append([
            title,
            f"A{res.agent_id}/I{res.iteration_id}",
            f"{m.accuracy:.3f}",
            f"{m.coverage:.3f}",
            f"{m.compression_ratio:.3f}",
            f"{m.num_reprs}",
            f"{m.avg_processing_time:.3f}s",
            f"{m.total_points}",
            f"{m.correct_classifications}/{m.confident_classifications}"
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=['Rank', 'Agent/Iter', 'Acc', 'Cov', 'Comp', 'Reprs', 'Time', 'Points', 'Correct/Conf'],
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.15, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Color rows
    for i in range(len(table_data)):
        color = '#90EE90' if i < 3 else '#FFB6C1'
        for j in range(len(table_data[0])):
            table[(i+1, j)].set_facecolor(color)

    plt.suptitle('Detailed analysis of best and worst results',
                fontsize=14, fontweight='bold', y=0.98)
    save_plot(fig, save_dir, 'best_worst_complete.png')
