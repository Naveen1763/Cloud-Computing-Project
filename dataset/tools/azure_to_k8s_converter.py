#!/usr/bin/env python3
"""
Azure to Kubernetes Workload Dataset Converter

This script converts Azure Functions traces to Kubernetes-compatible workload format.
It performs feature extraction, archetype classification, and resource estimation.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.data.features import extract_time_series_features
    from src.models.labeling import apply_weak_supervision
except ImportError:
    print("Warning: Could not import from src. Using simplified feature extraction.")
    
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureToK8sConverter:
    """Converts Azure Functions traces to Kubernetes workload format"""
    
    def __init__(self, azure_data_path: str, output_path: str):
        self.azure_data_path = Path(azure_data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_path / 'workloads').mkdir(exist_ok=True)
        (self.output_path / 'archetypes').mkdir(exist_ok=True)
        
        # Archetype definitions
        self.archetype_criteria = {
            'spike': lambda f: f['kurtosis'] > 20 and f['peak_to_mean_ratio'] > 50,
            'periodic': lambda f: f['spectral_entropy'] < 0.4 and f['autocorr_lag_60'] > 0.6,
            'ramp': lambda f: abs(f['trend_strength']) > 0.1,
            'stationary_noisy': lambda f: f['variance_rps'] > 0.5 and abs(f['trend_strength']) < 0.05
        }
        
    def load_azure_traces(self, days: int = 14) -> Dict[str, np.ndarray]:
        """Load Azure Functions invocation traces"""
        logger.info(f"Loading Azure traces for {days} days...")
        
        traces = {}
        for day in range(1, days + 1):
            filename = f"invocations_per_function_md.anon.d{day:02d}.csv"
            filepath = self.azure_data_path / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
                
            logger.info(f"Loading day {day}...")
            df = pd.read_csv(filepath)
            
            # Group by function and extract time series
            for _, row in df.iterrows():
                func_id = row['HashFunction']
                # Get minute columns (1-1440)
                minute_data = [row[str(i)] for i in range(1, 1441)]
                
                if func_id not in traces:
                    traces[func_id] = []
                traces[func_id].extend(minute_data)
                
        logger.info(f"Loaded {len(traces)} unique functions")
        return traces
    
    def extract_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """Extract features from time series"""
        features = {}
        
        # Basic statistics
        features['mean_rps'] = np.mean(time_series) / 60.0  # Convert to RPS
        features['variance_rps'] = np.var(time_series) / 3600.0
        features['std_dev_rps'] = np.std(time_series) / 60.0
        features['skewness'] = stats.skew(time_series)
        features['kurtosis'] = stats.kurtosis(time_series)
        
        # Peak analysis
        peaks, _ = find_peaks(time_series, height=np.mean(time_series))
        features['num_peaks'] = len(peaks)
        features['peak_to_mean_ratio'] = (np.max(time_series) / np.mean(time_series) 
                                         if np.mean(time_series) > 0 else 0)
        
        # Autocorrelation
        if len(time_series) > 120:
            features['autocorr_lag_60'] = np.corrcoef(time_series[:-60], time_series[60:])[0, 1]
            features['autocorr_lag_120'] = np.corrcoef(time_series[:-120], time_series[120:])[0, 1]
        else:
            features['autocorr_lag_60'] = 0
            features['autocorr_lag_120'] = 0
            
        # Spectral entropy (simplified)
        # In practice, you'd use proper spectral analysis
        features['spectral_entropy'] = stats.entropy(np.histogram(time_series, bins=20)[0] + 1e-10)
        
        # Trend analysis (simplified linear trend)
        x = np.arange(len(time_series))
        slope, _, _, _, _ = stats.linregress(x, time_series)
        features['trend_strength'] = slope / (np.std(time_series) + 1e-10)
        
        # Seasonality (simplified - ratio of autocorrelation at different lags)
        features['seasonality_strength'] = abs(features['autocorr_lag_60'])
        
        return features
    
    def classify_archetype(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify workload archetype based on features"""
        scores = {}
        
        for archetype, criterion in self.archetype_criteria.items():
            try:
                if criterion(features):
                    scores[archetype] = 1.0
                else:
                    scores[archetype] = 0.0
            except:
                scores[archetype] = 0.0
                
        # If multiple matches, choose based on priority
        if scores['spike'] > 0:
            return 'spike', 0.85
        elif scores['periodic'] > 0:
            return 'periodic', 0.90
        elif scores['ramp'] > 0:
            return 'ramp', 0.80
        else:
            return 'stationary_noisy', 0.75
            
    def estimate_resources(self, time_series: np.ndarray, 
                          execution_time_ms: float = 100) -> Dict[str, Dict[str, int]]:
        """Estimate CPU and memory requirements"""
        # Convert to requests per second
        rps_series = time_series / 60.0
        
        # CPU estimation (millicores)
        # Formula: cpu = (execution_time_ms * rps * 1000) / target_utilization
        target_utilization = 0.7
        cpu_series = (execution_time_ms * rps_series * 1000) / target_utilization
        
        cpu_requirements = {
            'p50': int(np.percentile(cpu_series, 50)),
            'p90': int(np.percentile(cpu_series, 90)),
            'p99': int(np.percentile(cpu_series, 99))
        }
        
        # Memory estimation (MB)
        # Simplified: based on request rate with minimum baseline
        memory_base = 128
        memory_per_rps = 50
        memory_series = memory_base + (rps_series * memory_per_rps)
        
        memory_requirements = {
            'p50': int(np.percentile(memory_series, 50)),
            'p90': int(np.percentile(memory_series, 90)),
            'p99': int(np.percentile(memory_series, 99))
        }
        
        return {
            'cpu_millicores': cpu_requirements,
            'memory_mb': memory_requirements
        }
    
    def create_workload_json(self, workload_id: str, time_series: np.ndarray,
                           start_date: datetime) -> Dict[str, Any]:
        """Create a complete workload JSON object"""
        # Extract features
        features = self.extract_features(time_series)
        
        # Classify archetype
        archetype, confidence = self.classify_archetype(features)
        
        # Estimate resources
        resources = self.estimate_resources(time_series)
        
        # Generate timestamps
        timestamps = []
        for i in range(len(time_series)):
            ts = start_date + timedelta(minutes=i)
            timestamps.append(ts.isoformat() + 'Z')
            
        # Convert to RPS
        rps_series = (time_series / 60.0).tolist()
        
        # Create workload object
        workload = {
            'workload_id': workload_id,
            'archetype': archetype,
            'archetype_confidence': confidence,
            'duration_minutes': len(time_series),
            'resource_requirements': resources,
            'slo_targets': {
                'response_time_ms': {
                    'p50': 100,
                    'p90': 200,
                    'p99': 500
                }
            },
            'request_trace': {
                'timestamps': timestamps,
                'requests_per_minute': time_series.tolist(),
                'requests_per_second': rps_series
            },
            'features': features
        }
        
        return workload
    
    def convert_dataset(self, max_workloads: int = None):
        """Convert entire dataset"""
        logger.info("Starting dataset conversion...")
        
        # Load traces
        traces = self.load_azure_traces()
        
        if max_workloads:
            trace_items = list(traces.items())[:max_workloads]
        else:
            trace_items = list(traces.items())
            
        # Track archetypes
        archetypes = {
            'spike': [],
            'periodic': [],
            'ramp': [],
            'stationary_noisy': []
        }
        
        # Convert each trace
        start_date = datetime(2019, 7, 1)
        
        for idx, (func_id, time_series) in enumerate(trace_items):
            if idx % 100 == 0:
                logger.info(f"Processing workload {idx + 1}/{len(trace_items)}")
                
            # Skip if too short
            if len(time_series) < 1440:  # Less than 1 day
                continue
                
            # Create workload
            workload_id = f"w_{idx + 1:04d}"
            time_series_np = np.array(time_series)
            
            try:
                workload = self.create_workload_json(workload_id, time_series_np, start_date)
                
                # Save workload
                output_file = self.output_path / 'workloads' / f"{workload_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(workload, f, indent=2)
                    
                # Track archetype
                archetypes[workload['archetype']].append(workload_id)
                
            except Exception as e:
                logger.error(f"Error processing {func_id}: {e}")
                continue
                
        # Save archetype mappings
        for archetype, workload_ids in archetypes.items():
            output_file = self.output_path / 'archetypes' / f"{archetype}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'archetype': archetype,
                    'count': len(workload_ids),
                    'workload_ids': workload_ids
                }, f, indent=2)
                
        logger.info(f"Conversion complete! Generated {idx + 1} workloads")
        logger.info(f"Archetype distribution: {[(k, len(v)) for k, v in archetypes.items()]}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Azure traces to K8s dataset')
    parser.add_argument('--azure-path', type=str, required=True,
                       help='Path to Azure Functions dataset')
    parser.add_argument('--output-path', type=str, 
                       default='../dataset',
                       help='Output path for K8s dataset')
    parser.add_argument('--max-workloads', type=int,
                       help='Maximum number of workloads to convert')
    
    args = parser.parse_args()
    
    converter = AzureToK8sConverter(args.azure_path, args.output_path)
    converter.convert_dataset(args.max_workloads)


if __name__ == '__main__':
    main()