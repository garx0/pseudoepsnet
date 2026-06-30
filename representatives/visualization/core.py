"""Base utilities for visualization"""
import matplotlib
matplotlib.use('Agg') # Agg backend without GUI

import matplotlib.pyplot as plt
import os
from typing import Optional

def setup_plot_style():
    """Setup plot style"""
    plt.style.use('seaborn-v0_8-darkgrid')

def save_plot(fig, save_dir, filename):
    """Save plot"""
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def get_figure(rows: int = 1, cols: int = 1, figsize: tuple = None):
    """Create figure with default sizes"""
    if figsize is None:
        figsize = (18, 6) if cols == 3 else (15, 10) if rows == 2 else (12, 5)
    return plt.subplots(rows, cols, figsize=figsize)
