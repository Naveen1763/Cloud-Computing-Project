#!/usr/bin/env python3
"""
Example script for creating visualizations of results.

This creates the key figures from the paper:
1. Performance vs Cost tradeoff scatter plot
2. Time series comparison
3. REI comparison
4. Confusion matrix heatmap
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_performance_cost_scatter(results_df: pd.DataFrame, output_path: Path):
    """Create performance vs cost scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors and markers for each autoscaler
    styles = {
        'AAPA': {'color': 'blue', 'marker': 'o', 'label': 'AAPA'},
        'HPA': {'color': 'red', 'marker': 's', 'label': 'Kubernetes HPA'},
        'Generic_Predictive': {'color': 'green', 'marker': '^', 'label': 'Generic Predictive'}
    }
    
    # Plot each autoscaler
    for autoscaler, style in styles.items():
        data = results_df[results_df['autoscaler'] == autoscaler]
        ax.scatter(
            data['total_pod_minutes'],
            data['slo_violation_rate'] * 100,
            c=style['color'],
            marker=style['marker'],
            label=style['label'],
            alpha=0.6,
            s=50
        )
    
    # Add ideal point
    ax.scatter([0], [0], marker='*', s=200, c='gold', label='Ideal', zorder=5)
    
    # Labels and formatting
    ax.set_xlabel('Resource Usage (Pod-Minutes)', fontsize=12)
    ax.set_ylabel('SLO Violation Rate (%)', fontsize=12)
    ax.set_title('Performance vs. Cost Tradeoff', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Lower is better →', xy=(0.7, 0.95), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic')
    ax.annotate('↓ Lower is better', xy=(0.05, 0.5), xycoords='axes fraction',
                ha='center', rotation=90, fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance-cost scatter to {output_path}")


def create_time_series_comparison(workload_data: dict, results: dict, output_path: Path):
    """Create time series comparison plot."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    time_minutes = np.arange(len(workload_data['requests']))
    
    # Plot 1: Request rate
    axes[0].plot(time_minutes, workload_data['requests'], 'k-', linewidth=1.5)
    axes[0].set_ylabel('Requests/min')
    axes[0].set_title('Workload Pattern')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Replica count comparison
    axes[1].plot(time_minutes, results['aapa']['replicas'], 'b-', 
                 label='AAPA', linewidth=2)
    axes[1].plot(time_minutes, results['hpa']['replicas'], 'r--', 
                 label='HPA', linewidth=2)
    axes[1].set_ylabel('Active Replicas')
    axes[1].set_title('Autoscaler Response')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Response time comparison
    axes[2].plot(time_minutes, results['aapa']['response_times'], 'b-', 
                 label='AAPA', linewidth=1.5, alpha=0.7)
    axes[2].plot(time_minutes, results['hpa']['response_times'], 'r--', 
                 label='HPA', linewidth=1.5, alpha=0.7)
    axes[2].axhline(y=500, color='k', linestyle=':', label='SLO (500ms)')
    axes[2].set_ylabel('Response Time (ms)')
    axes[2].set_xlabel('Time (minutes)')
    axes[2].set_title('Performance Impact')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved time series comparison to {output_path}")


def create_rei_comparison(results_df: pd.DataFrame, output_path: Path):
    """Create REI (Resource Efficiency Index) comparison bar plot."""
    # Calculate REI for each autoscaler and workload type
    rei_data = []
    
    for autoscaler in results_df['autoscaler'].unique():
        for workload_type in ['SPIKE', 'PERIODIC', 'RAMP', 'STATIONARY_NOISY']:
            data = results_df[
                (results_df['autoscaler'] == autoscaler) & 
                (results_df['workload_type'] == workload_type)
            ]
            
            if len(data) > 0:
                # Calculate REI components
                slo_score = 1 - data['slo_violation_rate'].mean()
                efficiency_score = 1 / (1 + data['total_pod_minutes'].mean() / 100)
                stability_score = 1 / (1 + data['scaling_actions'].mean() / 10)
                
                # Weighted REI
                rei = 0.5 * slo_score + 0.3 * efficiency_score + 0.2 * stability_score
                
                rei_data.append({
                    'autoscaler': autoscaler,
                    'workload_type': workload_type,
                    'rei': rei
                })
    
    rei_df = pd.DataFrame(rei_data)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the fixed archetype order; an archetype absent from the results
    # simply gets a zero-height bar.
    archetypes = ['SPIKE', 'PERIODIC', 'RAMP', 'STATIONARY_NOISY']
    x = np.arange(len(archetypes))
    width = 0.25

    for i, autoscaler in enumerate(['AAPA', 'HPA', 'Generic_Predictive']):
        data = rei_df[rei_df['autoscaler'] == autoscaler]
        values = [data[data['workload_type'] == wt]['rei'].values[0]
                 if len(data[data['workload_type'] == wt]) > 0 else 0
                 for wt in archetypes]

        ax.bar(x + i * width, values, width, label=autoscaler)
    
    ax.set_xlabel('Workload Type', fontsize=12)
    ax.set_ylabel('Resource Efficiency Index (REI)', fontsize=12)
    ax.set_title('REI Comparison by Workload Type', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['SPIKE', 'PERIODIC', 'RAMP', 'STATIONARY'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved REI comparison to {output_path}")


def create_confusion_matrix_heatmap(cm: np.ndarray, output_path: Path, labels=None):
    """Create confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is None:
        labels = ['PERIODIC', 'SPIKE', 'STATIONARY', 'RAMP']

    # The matrix only covers classes present in the test split, so trim or pad
    # the tick labels to match rather than letting seaborn mis-align them.
    n = cm.shape[0]
    labels = list(labels)[:n]
    labels += [f'CLASS_{i}' for i in range(len(labels), n)]

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Workload Classification Confusion Matrix', fontsize=14)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}', 
            transform=ax.transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def main():
    """Create all visualizations."""
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Check if results exist
    results_path = results_dir / "simulation_results.csv"
    if not results_path.exists():
        logger.error(f"Results not found at {results_path}")
        logger.error("Please run run_simulation.py first")
        return
    
    # Load results
    logger.info("Loading simulation results...")
    results_df = pd.read_csv(results_path)
    
    # run_simulation.py writes the ground-truth archetype per workload. Only
    # fall back to synthetic labels for older result files that lack it.
    if 'workload_type' not in results_df.columns:
        logger.warning("No workload_type column found - using synthetic labels")
        np.random.seed(42)
        results_df['workload_type'] = np.random.choice(
            ['SPIKE', 'PERIODIC', 'RAMP', 'STATIONARY_NOISY'],
            size=len(results_df),
            p=[0.25, 0.4, 0.1, 0.25]
        )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Performance vs Cost scatter
    create_performance_cost_scatter(
        results_df,
        figures_dir / "performance_cost_tradeoff.png"
    )
    
    # 2. Time series comparison (using mock data for example)
    mock_workload = {
        'requests': np.concatenate([
            np.random.poisson(10, 500),
            np.random.poisson(100, 100),  # Spike
            np.random.poisson(20, 840)
        ])
    }
    mock_results = {
        'aapa': {
            'replicas': np.concatenate([
                np.ones(500) * 2,
                np.linspace(2, 10, 100),
                np.ones(840) * 3
            ]),
            'response_times': np.concatenate([
                np.random.normal(100, 20, 500),
                np.random.normal(200, 50, 100),
                np.random.normal(120, 30, 840)
            ])
        },
        'hpa': {
            'replicas': np.concatenate([
                np.ones(600) * 1,
                np.linspace(1, 8, 100),
                np.ones(740) * 2
            ]),
            'response_times': np.concatenate([
                np.random.normal(150, 30, 500),
                np.random.normal(600, 100, 100),
                np.random.normal(200, 40, 840)
            ])
        }
    }
    create_time_series_comparison(
        mock_workload,
        mock_results,
        figures_dir / "time_series_comparison.png"
    )
    
    # 3. REI comparison
    create_rei_comparison(
        results_df,
        figures_dir / "rei_comparison.png"
    )
    
    # 4. Confusion matrix from the trained classifier, if available
    cm_path = Path("models") / "confusion_matrix.npz"
    if cm_path.exists():
        payload = np.load(cm_path, allow_pickle=True)
        cm = payload['cm']
        cm_labels = [str(label) for label in payload['labels']]
    else:
        logger.warning(f"{cm_path} not found - using illustrative confusion matrix")
        cm = np.array([
            [20998, 12, 5, 0],
            [8, 5269, 3, 0],
            [6, 4, 3590, 0],
            [0, 2, 1, 57]
        ])
        cm_labels = ['PERIODIC', 'SPIKE', 'STATIONARY', 'RAMP']

    create_confusion_matrix_heatmap(
        cm,
        figures_dir / "confusion_matrix.png",
        labels=cm_labels
    )
    
    logger.info(f"All visualizations saved to {figures_dir}")
    logger.info("Visualization creation completed!")


if __name__ == "__main__":
    main()