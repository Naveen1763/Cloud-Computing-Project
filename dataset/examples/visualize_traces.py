#!/usr/bin/env python3
"""
Example: Visualizing workload traces

This script demonstrates how to create visualizations of workload patterns
from the dataset.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_workload(workload_id: str, dataset_path: str = '../workloads') -> dict:
    """Load a workload by ID"""
    filepath = Path(dataset_path) / f'{workload_id}.json'
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_request_trace(workload: dict, hours: int = 24, save_path: str = None):
    """Plot request trace for specified hours"""
    # Get data
    rpm = workload['request_trace']['requests_per_minute']
    rps = workload['request_trace']['requests_per_second']
    
    # Limit to specified hours
    minutes = min(hours * 60, len(rpm))
    rpm = rpm[:minutes]
    rps = rps[:minutes]
    
    # Create time axis
    time_axis = np.arange(minutes)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot requests per minute
    ax1.plot(time_axis, rpm, 'b-', linewidth=0.8)
    ax1.set_ylabel('Requests per Minute')
    ax1.set_title(f"Workload {workload['workload_id']} - {workload['archetype'].upper()} Pattern")
    ax1.grid(True, alpha=0.3)
    
    # Add mean line
    mean_rpm = np.mean(rpm)
    ax1.axhline(y=mean_rpm, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_rpm:.1f}')
    ax1.legend()
    
    # Plot requests per second
    ax2.plot(time_axis, rps, 'g-', linewidth=0.8)
    ax2.set_ylabel('Requests per Second')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    mean_rps = np.mean(rps)
    ax2.axhline(y=mean_rps, color='r', linestyle='--', alpha=0.7,
                label=f'Mean: {mean_rps:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_archetype_comparison(dataset_path: str = '..', save_path: str = None):
    """Plot representative traces for each archetype"""
    archetypes = ['spike', 'periodic', 'ramp', 'stationary_noisy']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, archetype in enumerate(archetypes):
        try:
            # Load archetype file
            with open(Path(dataset_path) / 'archetypes' / f'{archetype}.json', 'r') as f:
                arch_data = json.load(f)
            
            if arch_data['workload_ids']:
                # Load first workload of this type
                workload_id = arch_data['workload_ids'][0]
                workload = load_workload(workload_id, Path(dataset_path) / 'workloads')
                
                # Plot first 2 hours
                minutes = 120
                rpm = workload['request_trace']['requests_per_minute'][:minutes]
                time_axis = np.arange(len(rpm))
                
                ax = axes[idx]
                ax.plot(time_axis, rpm, linewidth=1)
                ax.set_title(f"{archetype.upper()} Pattern", fontsize=12, fontweight='bold')
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Requests per Minute')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                ax.text(0.02, 0.98, 
                       f"Mean: {np.mean(rpm):.1f}\nMax: {np.max(rpm):.1f}\nStd: {np.std(rpm):.1f}",
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except Exception as e:
            print(f"Error plotting {archetype}: {e}")
    
    plt.suptitle('Workload Archetype Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_resource_distribution(workload: dict, save_path: str = None):
    """Plot resource requirement distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # CPU distribution
    cpu = workload['resource_requirements']['cpu_millicores']
    percentiles = ['p50', 'p90', 'p99']
    values = [cpu[p] for p in percentiles]
    
    ax1.bar(percentiles, values, color=['green', 'yellow', 'red'], alpha=0.7)
    ax1.set_ylabel('CPU (millicores)')
    ax1.set_title('CPU Requirements by Percentile')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(values):
        ax1.text(i, v + 10, str(v), ha='center', va='bottom')
    
    # Memory distribution
    mem = workload['resource_requirements']['memory_mb']
    values = [mem[p] for p in percentiles]
    
    ax2.bar(percentiles, values, color=['green', 'yellow', 'red'], alpha=0.7)
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Requirements by Percentile')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(values):
        ax2.text(i, v + 5, str(v), ha='center', va='bottom')
    
    plt.suptitle(f"Resource Requirements - {workload['workload_id']} ({workload['archetype'].upper()})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Resource plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Example visualizations"""
    print("Creating example visualizations...")
    
    # Example 1: Plot a spike workload
    print("\n1. Plotting spike workload pattern...")
    try:
        workload = load_workload('w_0001')
        plot_request_trace(workload, hours=2, save_path='spike_pattern.png')
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Plot archetype comparison
    print("\n2. Creating archetype comparison plot...")
    try:
        plot_archetype_comparison(save_path='archetype_comparison.png')
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Plot resource requirements
    print("\n3. Plotting resource requirements...")
    try:
        workload = load_workload('w_0002')
        plot_resource_distribution(workload, save_path='resource_requirements.png')
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nVisualization examples complete!")


if __name__ == '__main__':
    main()