"""
ARC Prize 2025 - Visualization Module

This module provides utilities for visualizing ARC-AGI grids.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
from typing import List


# ARC color palette (standard 10 colors)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Light Blue
    '#870C25',  # 9: Dark Red
]

ARC_CMAP = ListedColormap(ARC_COLORS)


def plot_grid(grid: List[List[int]], ax=None, title: str = ""):
    """
    Plot a single ARC grid.

    Args:
        grid: 2D list representing the grid
        ax: Matplotlib axis (creates new if None)
        title: Title for the plot
    """
    grid_array = np.array(grid)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Plot the grid
    ax.imshow(grid_array, cmap=ARC_CMAP, vmin=0, vmax=9)

    # Add grid lines
    height, width = grid_array.shape
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Add dimensions
    ax.text(0.02, 0.98, f'{height}×{width}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)


def plot_task(challenge: dict, solution: List = None, task_id: str = ""):
    """
    Plot a complete ARC task (training examples + test).

    Args:
        challenge: Challenge dictionary
        solution: Solution list (optional)
        task_id: Task identifier for title
    """
    train_examples = challenge['train']
    test_examples = challenge['test']

    n_train = len(train_examples)
    n_test = len(test_examples)
    n_rows = n_train + n_test

    # Create figure
    fig, axes = plt.subplots(n_rows, 2 + (1 if solution else 0),
                             figsize=(8 * (2 + (1 if solution else 0)), 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot training examples
    for i, example in enumerate(train_examples):
        plot_grid(example['input'], axes[i, 0], f"Train {i+1} - Input")
        plot_grid(example['output'], axes[i, 1], f"Train {i+1} - Output")
        if solution:
            axes[i, 2].axis('off')

    # Plot test examples
    for i, example in enumerate(test_examples):
        row_idx = n_train + i
        plot_grid(example['input'], axes[row_idx, 0], f"Test {i+1} - Input")

        if solution and i < len(solution):
            plot_grid(solution[i], axes[row_idx, 1], f"Test {i+1} - Expected")
        else:
            axes[row_idx, 1].text(0.5, 0.5, "?",
                                 ha='center', va='center',
                                 fontsize=48, color='gray',
                                 transform=axes[row_idx, 1].transAxes)
            axes[row_idx, 1].set_xticks([])
            axes[row_idx, 1].set_yticks([])
            axes[row_idx, 1].set_title(f"Test {i+1} - Output", fontsize=12, fontweight='bold')

        if solution:
            axes[row_idx, 2].axis('off')

    # Add main title
    fig.suptitle(f'Task: {task_id}' if task_id else 'ARC Task',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    return fig


def plot_comparison(input_grid: List[List[int]],
                   predicted_grid: List[List[int]],
                   expected_grid: List[List[int]] = None):
    """
    Plot input, predicted, and optionally expected output side by side.

    Args:
        input_grid: Input grid
        predicted_grid: Predicted output grid
        expected_grid: Expected output grid (optional)
    """
    n_cols = 3 if expected_grid else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    if n_cols == 2:
        axes = [axes[0], axes[1]]

    plot_grid(input_grid, axes[0], "Input")
    plot_grid(predicted_grid, axes[1], "Predicted")

    if expected_grid:
        plot_grid(expected_grid, axes[2], "Expected")

    plt.tight_layout()
    return fig


def plot_color_distribution(grid: List[List[int]], ax=None, title: str = ""):
    """
    Plot color distribution in a grid.

    Args:
        grid: 2D list representing the grid
        ax: Matplotlib axis (creates new if None)
        title: Title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Count colors
    grid_array = np.array(grid)
    unique, counts = np.unique(grid_array, return_counts=True)

    # Plot bar chart
    colors = [ARC_COLORS[i] for i in unique]
    ax.bar(unique, counts, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Color', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(unique)
    ax.set_title(title if title else 'Color Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


if __name__ == "__main__":
    # Example usage
    from data_loader import ARCDataLoader, print_task_info

    loader = ARCDataLoader()
    training_ids = loader.get_task_ids("training")

    if training_ids:
        # Load and visualize first task
        task_id = training_ids[0]
        challenge, solution = loader.load_task(task_id, "training")

        print_task_info(task_id, challenge, solution)

        # Plot the task
        fig = plot_task(challenge, solution, task_id)
        plt.show()
