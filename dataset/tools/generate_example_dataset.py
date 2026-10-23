#!/usr/bin/env python3
"""
Generate example workload dataset with full traces

This script generates a small example dataset with complete 14-day traces
for demonstration purposes.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_spike_pattern(duration_minutes: int = 20160) -> np.ndarray:
    """Generate a spike pattern workload"""
    # Base low activity
    base_load = np.random.poisson(lam=5, size=duration_minutes)
    
    # Add spikes
    spike_times = np.random.choice(duration_minutes, size=20, replace=False)
    for spike_time in spike_times:
        spike_duration = np.random.randint(5, 30)
        spike_intensity = np.random.randint(100, 500)
        start = max(0, spike_time)
        end = min(duration_minutes, spike_time + spike_duration)
        base_load[start:end] += spike_intensity
    
    return base_load


def generate_periodic_pattern(duration_minutes: int = 20160) -> np.ndarray:
    """Generate a periodic pattern workload"""
    time = np.arange(duration_minutes)
    
    # Daily pattern (1440 minutes = 1 day)
    daily_pattern = 50 + 30 * np.sin(2 * np.pi * time / 1440)
    
    # Weekly pattern
    weekly_pattern = 10 * np.sin(2 * np.pi * time / (1440 * 7))
    
    # Add some noise
    noise = np.random.normal(0, 5, duration_minutes)
    
    pattern = daily_pattern + weekly_pattern + noise
    pattern = np.maximum(pattern, 0)  # Ensure non-negative
    
    return pattern.astype(int)


def generate_ramp_pattern(duration_minutes: int = 20160) -> np.ndarray:
    """Generate a ramp pattern workload"""
    # Linear increase over time
    time = np.arange(duration_minutes)
    
    # Multiple ramp segments
    pattern = np.zeros(duration_minutes)
    segments = 4
    segment_length = duration_minutes // segments
    
    for i in range(segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, duration_minutes)
        
        if i % 2 == 0:
            # Ramp up
            pattern[start:end] = np.linspace(20, 100, end - start)
        else:
            # Ramp down
            pattern[start:end] = np.linspace(100, 20, end - start)
    
    # Add some noise
    noise = np.random.normal(0, 5, duration_minutes)
    pattern = pattern + noise
    pattern = np.maximum(pattern, 0)
    
    return pattern.astype(int)


def generate_stationary_noisy_pattern(duration_minutes: int = 20160) -> np.ndarray:
    """Generate a stationary noisy pattern workload"""
    # Constant base with random noise
    base_load = 50
    noise = np.random.normal(0, 15, duration_minutes)
    pattern = base_load + noise
    
    # Add occasional random fluctuations
    fluctuation_times = np.random.choice(duration_minutes, size=100, replace=False)
    for t in fluctuation_times:
        pattern[t] += np.random.normal(0, 30)
    
    pattern = np.maximum(pattern, 0)
    return pattern.astype(int)


def extract_features(time_series: np.ndarray) -> dict:
    """Extract features from time series"""
    from scipy import stats
    from scipy.signal import find_peaks
    
    features = {}
    
    # Basic statistics
    features['mean_rps'] = float(np.mean(time_series) / 60.0)
    features['variance_rps'] = float(np.var(time_series) / 3600.0)
    features['std_dev_rps'] = float(np.std(time_series) / 60.0)
    features['skewness'] = float(stats.skew(time_series))
    features['kurtosis'] = float(stats.kurtosis(time_series))
    
    # Peak analysis
    peaks, _ = find_peaks(time_series, height=np.mean(time_series))
    features['num_peaks'] = int(len(peaks))
    features['peak_to_mean_ratio'] = float(np.max(time_series) / np.mean(time_series) if np.mean(time_series) > 0 else 0)
    
    # Autocorrelation
    if len(time_series) > 120:
        features['autocorr_lag_60'] = float(np.corrcoef(time_series[:-60], time_series[60:])[0, 1])
        features['autocorr_lag_120'] = float(np.corrcoef(time_series[:-120], time_series[120:])[0, 1])
    else:
        features['autocorr_lag_60'] = 0.0
        features['autocorr_lag_120'] = 0.0
    
    # Simplified spectral entropy
    hist, _ = np.histogram(time_series, bins=20)
    hist = hist + 1e-10
    features['spectral_entropy'] = float(stats.entropy(hist))
    
    # Trend
    x = np.arange(len(time_series))
    slope, _, _, _, _ = stats.linregress(x, time_series)
    features['trend_strength'] = float(slope / (np.std(time_series) + 1e-10))
    features['seasonality_strength'] = float(abs(features['autocorr_lag_60']))
    
    return features


def create_workload(workload_id: str, archetype: str, pattern_func, confidence: float) -> dict:
    """Create a complete workload JSON object"""
    # Generate 14-day trace
    time_series = pattern_func(20160)  # 14 days * 1440 minutes/day
    
    # Extract features
    features = extract_features(time_series)
    
    # Estimate resources based on load
    rps_series = time_series / 60.0
    cpu_series = (100 * rps_series * 1000) / 0.7  # 100ms exec time, 70% utilization
    memory_series = 128 + (rps_series * 50)
    
    # Generate timestamps (only first 100 for example, full would be 20160)
    start_date = datetime(2019, 7, 1)
    timestamps = []
    for i in range(min(100, len(time_series))):  # Limited for file size
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
                'p50': int(np.percentile(cpu_series, 50)),
                'p90': int(np.percentile(cpu_series, 90)),
                'p99': int(np.percentile(cpu_series, 99))
            },
            'memory_mb': {
                'p50': int(np.percentile(memory_series, 50)),
                'p90': int(np.percentile(memory_series, 90)),
                'p99': int(np.percentile(memory_series, 99))
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
            'requests_per_minute': time_series[:100].tolist(),  # First 100 minutes as example
            'requests_per_second': (time_series[:100] / 60.0).tolist(),
            'total_minutes': len(time_series),
            'note': "Full trace contains 20160 minutes. Showing first 100 for example."
        },
        'features': features
    }
    
    return workload


def main():
    """Generate example dataset"""
    output_path = Path(__file__).parent.parent / 'workloads'
    output_path.mkdir(exist_ok=True)
    
    logger.info("Generating example workload dataset...")
    
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
        logger.info(f"Generating {workload_id} ({archetype})...")
        
        workload = create_workload(workload_id, archetype, pattern_func, confidence)
        
        # Save workload
        output_file = output_path / f"{workload_id}.json"
        with open(output_file, 'w') as f:
            json.dump(workload, f, indent=2)
        
        # Track archetype
        archetypes[archetype].append(workload_id)
    
    # Update archetype files
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
    
    logger.info(f"Generated {len(workloads_config)} example workloads")
    logger.info("Note: These are examples with truncated traces. Use azure_to_k8s_converter.py for full dataset generation.")


if __name__ == '__main__':
    main()