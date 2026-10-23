#!/usr/bin/env python3
"""
Example: Loading and exploring workload data

This script demonstrates how to load workload data from the dataset
and access various properties.
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add parent to path for any imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_workload(workload_id: str, dataset_path: str = '../workloads') -> dict:
    """Load a workload by ID"""
    filepath = Path(dataset_path) / f'{workload_id}.json'
    
    if not filepath.exists():
        raise FileNotFoundError(f"Workload {workload_id} not found at {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def print_workload_summary(workload: dict):
    """Print a summary of workload characteristics"""
    print(f"\n{'='*50}")
    print(f"Workload ID: {workload['workload_id']}")
    print(f"Archetype: {workload['archetype']} (confidence: {workload['archetype_confidence']:.2f})")
    print(f"Duration: {workload['duration_minutes']} minutes ({workload['duration_minutes']/1440:.1f} days)")
    
    # Resource requirements
    print(f"\nResource Requirements:")
    cpu = workload['resource_requirements']['cpu_millicores']
    print(f"  CPU (millicores): p50={cpu['p50']}, p90={cpu['p90']}, p99={cpu['p99']}")
    mem = workload['resource_requirements']['memory_mb']
    print(f"  Memory (MB): p50={mem['p50']}, p90={mem['p90']}, p99={mem['p99']}")
    
    # SLO targets
    print(f"\nSLO Targets:")
    slo = workload['slo_targets']['response_time_ms']
    print(f"  Response time (ms): p50={slo['p50']}, p90={slo['p90']}, p99={slo['p99']}")
    
    # Request trace stats
    rps = workload['request_trace']['requests_per_second']
    print(f"\nRequest Rate Statistics:")
    print(f"  Mean RPS: {np.mean(rps):.2f}")
    print(f"  Max RPS: {np.max(rps):.2f}")
    print(f"  Min RPS: {np.min(rps):.2f}")
    
    # Features
    print(f"\nKey Features:")
    features = workload['features']
    print(f"  Peak-to-mean ratio: {features['peak_to_mean_ratio']:.2f}")
    print(f"  Spectral entropy: {features['spectral_entropy']:.2f}")
    print(f"  Autocorrelation (60min): {features['autocorr_lag_60']:.2f}")
    print(f"{'='*50}\n")


def load_archetype_workloads(archetype: str, dataset_path: str = '..') -> list:
    """Load all workload IDs for a given archetype"""
    archetype_file = Path(dataset_path) / 'archetypes' / f'{archetype}.json'
    
    if not archetype_file.exists():
        raise FileNotFoundError(f"Archetype file not found: {archetype_file}")
    
    with open(archetype_file, 'r') as f:
        data = json.load(f)
    
    return data['workload_ids']


def main():
    """Example usage"""
    # Example 1: Load a single workload
    print("Example 1: Loading a single workload")
    try:
        workload = load_workload('w_0001')
        print_workload_summary(workload)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example 2: Load all spike workloads
    print("\nExample 2: Loading all spike workloads")
    try:
        spike_ids = load_archetype_workloads('spike')
        print(f"Found {len(spike_ids)} spike workloads: {spike_ids}")
        
        # Load and summarize first spike workload
        if spike_ids:
            workload = load_workload(spike_ids[0])
            print_workload_summary(workload)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    # Example 3: Accessing specific data
    print("\nExample 3: Accessing request trace data")
    try:
        workload = load_workload('w_0002')
        
        # Get first 10 minutes of data
        timestamps = workload['request_trace']['timestamps'][:10]
        rpm = workload['request_trace']['requests_per_minute'][:10]
        rps = workload['request_trace']['requests_per_second'][:10]
        
        print("First 10 minutes of trace data:")
        for i in range(10):
            print(f"  {timestamps[i]}: {rpm[i]} req/min ({rps[i]:.2f} req/sec)")
    except (FileNotFoundError, IndexError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()