#!/usr/bin/env python3
"""
Generate example workload dataset with full traces (simplified version)

This script generates a small example dataset with complete 14-day traces
for demonstration purposes, without external dependencies.
"""

import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

def generate_spike_pattern(duration_minutes: int = 20160) -> list:
    """Generate a spike pattern workload"""
    # Base low activity
    pattern = [random.randint(3, 8) for _ in range(duration_minutes)]
    
    # Add spikes
    num_spikes = 20
    for _ in range(num_spikes):
        spike_time = random.randint(0, duration_minutes - 30)
        spike_duration = random.randint(5, 30)
        spike_intensity = random.randint(100, 500)
        
        for i in range(spike_duration):
            if spike_time + i < duration_minutes:
                pattern[spike_time + i] += spike_intensity
    
    return pattern


def generate_periodic_pattern(duration_minutes: int = 20160) -> list:
    """Generate a periodic pattern workload"""
    pattern = []
    
    for i in range(duration_minutes):
        # Daily pattern (1440 minutes = 1 day)
        daily_value = 50 + 30 * math.sin(2 * math.pi * i / 1440)
        
        # Weekly pattern
        weekly_value = 10 * math.sin(2 * math.pi * i / (1440 * 7))
        
        # Add some noise
        noise = random.gauss(0, 5)
        
        value = daily_value + weekly_value + noise
        pattern.append(max(0, int(value)))
    
    return pattern


def generate_ramp_pattern(duration_minutes: int = 20160) -> list:
    """Generate a ramp pattern workload"""
    pattern = []
    segments = 4
    segment_length = duration_minutes // segments
    
    for segment in range(segments):
        if segment % 2 == 0:
            # Ramp up
            start_val = 20
            end_val = 100
        else:
            # Ramp down
            start_val = 100
            end_val = 20
        
        for i in range(segment_length):
            progress = i / segment_length
            value = start_val + (end_val - start_val) * progress
            noise = random.gauss(0, 5)
            pattern.append(max(0, int(value + noise)))
    
    # Fill any remaining minutes
    while len(pattern) < duration_minutes:
        pattern.append(pattern[-1])
    
    return pattern[:duration_minutes]


def generate_stationary_noisy_pattern(duration_minutes: int = 20160) -> list:
    """Generate a stationary noisy pattern workload"""
    base_load = 50
    pattern = []
    
    for i in range(duration_minutes):
        noise = random.gauss(0, 15)
        
        # Occasionally add larger fluctuations
        if random.random() < 0.005:  # 0.5% chance
            noise += random.gauss(0, 30)
        
        value = base_load + noise
        pattern.append(max(0, int(value)))
    
    return pattern


def calculate_simple_features(time_series: list) -> dict:
    """Calculate simple features without scipy"""
    features = {}
    
    # Basic statistics
    mean_val = sum(time_series) / len(time_series)
    features['mean_rps'] = mean_val / 60.0
    
    # Variance
    variance = sum((x - mean_val) ** 2 for x in time_series) / len(time_series)
    features['variance_rps'] = variance / 3600.0
    features['std_dev_rps'] = math.sqrt(variance) / 60.0
    
    # Simple peak detection
    peaks = 0
    for i in range(1, len(time_series) - 1):
        if time_series[i] > time_series[i-1] and time_series[i] > time_series[i+1]:
            if time_series[i] > mean_val * 1.5:
                peaks += 1
    features['num_peaks'] = peaks
    
    # Peak to mean ratio
    max_val = max(time_series)
    features['peak_to_mean_ratio'] = max_val / mean_val if mean_val > 0 else 0
    
    # Simple autocorrelation at lag 60
    if len(time_series) > 60:
        sum_product = 0
        sum_sq1 = 0
        sum_sq2 = 0
        for i in range(len(time_series) - 60):
            diff1 = time_series[i] - mean_val
            diff2 = time_series[i + 60] - mean_val
            sum_product += diff1 * diff2
            sum_sq1 += diff1 ** 2
            sum_sq2 += diff2 ** 2
        
        if sum_sq1 > 0 and sum_sq2 > 0:
            features['autocorr_lag_60'] = sum_product / math.sqrt(sum_sq1 * sum_sq2)
        else:
            features['autocorr_lag_60'] = 0
    else:
        features['autocorr_lag_60'] = 0
    
    features['autocorr_lag_120'] = 0  # Simplified
    
    # Simple entropy approximation
    features['spectral_entropy'] = 0.5  # Placeholder
    
    # Simple trend
    features['trend_strength'] = 0.01  # Placeholder
    features['seasonality_strength'] = abs(features['autocorr_lag_60'])
    
    # Approximate skewness and kurtosis
    features['skewness'] = 0.5  # Placeholder
    features['kurtosis'] = 3.0  # Placeholder
    
    return features


def create_workload(workload_id: str, archetype: str, pattern_func, confidence: float) -> dict:
    """Create a complete workload JSON object"""
    # Generate 14-day trace
    print(f"  Generating {archetype} pattern...")
    time_series = pattern_func(20160)  # 14 days * 1440 minutes/day
    
    # Extract features
    print(f"  Extracting features...")
    features = calculate_simple_features(time_series)
    
    # Estimate resources based on load
    rps_series = [x / 60.0 for x in time_series]
    cpu_series = [(100 * rps * 1000) / 0.7 for rps in rps_series]  # 100ms exec time, 70% utilization
    memory_series = [128 + (rps * 50) for rps in rps_series]
    
    # Calculate percentiles
    def percentile(data, p):
        sorted_data = sorted(data)
        index = int(len(sorted_data) * p / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    # Generate timestamps (only first 100 for example)
    start_date = datetime(2019, 7, 1)
    timestamps = []
    for i in range(min(100, len(time_series))):
        ts = start_date + timedelta(minutes=i)
        timestamps.append(ts.isoformat() + 'Z')
    
    # Create workload object
    workload = {
        'workload_id': workload_id,
        'archetype': archetype,
        'archetype_confidence': confidence,
        'duration_minutes': len(time_series),
        'resource_requirements': {
            'cpu_millicores': {
                'p50': int(percentile(cpu_series, 50)),
                'p90': int(percentile(cpu_series, 90)),
                'p99': int(percentile(cpu_series, 99))
            },
            'memory_mb': {
                'p50': int(percentile(memory_series, 50)),
                'p90': int(percentile(memory_series, 90)),
                'p99': int(percentile(memory_series, 99))
            }
        },
        'slo_targets': {
            'response_time_ms': {
                'p50': 100,
                'p90': 200,
                'p99': 500
            }
        },
        'request_trace': {
            'timestamps': timestamps,
            'requests_per_minute': time_series[:100],  # First 100 minutes as example
            'requests_per_second': [x / 60.0 for x in time_series[:100]],
            'total_minutes': len(time_series),
            'note': "Full trace contains 20160 minutes (14 days). Showing first 100 for example."
        },
        'features': features
    }
    
    return workload


def main():
    """Generate example dataset"""
    output_path = Path(__file__).parent.parent / 'workloads'
    output_path.mkdir(exist_ok=True)
    
    print("Generating example workload dataset...")
    
    # Define workloads to generate
    workloads_config = [
        ('w_0001', 'spike', generate_spike_pattern, 0.85),
        ('w_0002', 'periodic', generate_periodic_pattern, 0.90),
        ('w_0003', 'spike', generate_spike_pattern, 0.88),
        ('w_0004', 'ramp', generate_ramp_pattern, 0.80),
        ('w_0005', 'stationary_noisy', generate_stationary_noisy_pattern, 0.75),
    ]
    
    # Track archetypes
    archetypes = {
        'spike': [],
        'periodic': [],
        'ramp': [],
        'stationary_noisy': []
    }
    
    # Generate workloads
    for workload_id, archetype, pattern_func, confidence in workloads_config:
        print(f"\nGenerating {workload_id} ({archetype})...")
        
        workload = create_workload(workload_id, archetype, pattern_func, confidence)
        
        # Save workload
        output_file = output_path / f"{workload_id}.json"
        with open(output_file, 'w') as f:
            json.dump(workload, f, indent=2)
        print(f"  Saved to {output_file}")
        
        # Track archetype
        archetypes[archetype].append(workload_id)
    
    # Update archetype files
    print("\nUpdating archetype files...")
    archetype_path = Path(__file__).parent.parent / 'archetypes'
    for archetype, workload_ids in archetypes.items():
        if workload_ids:  # Only update if we have workloads
            output_file = archetype_path / f"{archetype}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'archetype': archetype,
                    'description': {
                        'spike': 'Workloads with sudden bursts and low baseline activity',
                        'periodic': 'Workloads with regular, predictable patterns',
                        'ramp': 'Workloads with gradual increases or decreases',
                        'stationary_noisy': 'Stable workloads with random noise'
                    }[archetype],
                    'count': len(workload_ids),
                    'workload_ids': workload_ids
                }, f, indent=2)
            print(f"  Updated {output_file}")
    
    print(f"\nGenerated {len(workloads_config)} example workloads")
    print("Note: These are examples with truncated traces (showing first 100 minutes).")
    print("Use azure_to_k8s_converter.py with real Azure data for full dataset generation.")


if __name__ == '__main__':
    main()