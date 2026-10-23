import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesFeatureExtractor:
    """Extract features from serverless workload time series."""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """Extract all features from a time series.
        
        Args:
            time_series: 1D array of invocation counts per minute
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Handle edge cases
        if len(time_series) == 0 or np.sum(time_series) == 0:
            return self._get_zero_features()
            
        # Statistical features
        features.update(self._extract_statistical_features(time_series))
        
        # Time domain features
        features.update(self._extract_time_domain_features(time_series))
        
        # Frequency domain features
        features.update(self._extract_frequency_features(time_series))
        
        # Activity pattern features
        features.update(self._extract_activity_features(time_series))
        
        self.feature_names = list(features.keys())
        return features
    
    def _extract_statistical_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        return {
            'mean': float(np.mean(ts)),
            'median': float(np.median(ts)),
            'std': float(np.std(ts)),
            'variance': float(np.var(ts)),
            'skewness': float(stats.skew(ts)),
            'kurtosis': float(stats.kurtosis(ts)),
            'min': float(np.min(ts)),
            'max': float(np.max(ts)),
            'range': float(np.max(ts) - np.min(ts)),
            'iqr': float(np.percentile(ts, 75) - np.percentile(ts, 25)),
            'cv': float(np.std(ts) / (np.mean(ts) + 1e-6)),  # Coefficient of variation
            'max_to_mean_ratio': float(np.max(ts) / (np.mean(ts) + 1e-6)),
            'max_to_median_ratio': float(np.max(ts) / (np.median(ts) + 1e-6))
        }
    
    def _extract_time_domain_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract time-based features."""
        features = {}
        
        # Autocorrelation at different lags
        for lag in [1, 5, 10, 30, 60, 120, 1440]:  # 1min, 5min, 10min, 30min, 1hr, 2hr, 1day
            if lag < len(ts):
                features[f'autocorr_lag_{lag}'] = float(self._autocorrelation(ts, lag))
            else:
                features[f'autocorr_lag_{lag}'] = 0.0
        
        # Trend features
        time_indices = np.arange(len(ts))
        slope, intercept, r_value, _, _ = stats.linregress(time_indices, ts)
        features['linear_trend_slope'] = float(slope)
        features['linear_trend_r2'] = float(r_value ** 2)
        
        # Change statistics
        diff = np.diff(ts)
        if len(diff) > 0:
            features['mean_abs_change'] = float(np.mean(np.abs(diff)))
            features['mean_change'] = float(np.mean(diff))
            features['std_change'] = float(np.std(diff))
        else:
            features['mean_abs_change'] = 0.0
            features['mean_change'] = 0.0
            features['std_change'] = 0.0
        
        # Peak detection
        mean_ts = np.mean(ts)
        std_ts = np.std(ts)
        if std_ts > 0:
            peaks, properties = signal.find_peaks(ts, height=mean_ts + std_ts)
            features['num_peaks'] = float(len(peaks))
            features['peaks_per_hour'] = float(len(peaks) / (len(ts) / 60))
        else:
            features['num_peaks'] = 0.0
            features['peaks_per_hour'] = 0.0
        
        return features
    
    def _extract_frequency_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features."""
        features = {}
        
        # Compute FFT
        fft_vals = np.abs(fft(ts))
        fft_vals = fft_vals[:len(fft_vals)//2]  # Keep only positive frequencies
        
        # Normalize
        if np.sum(fft_vals) > 0:
            fft_vals = fft_vals / np.sum(fft_vals)
        
        # Spectral entropy
        features['spectral_entropy'] = float(self._spectral_entropy(fft_vals))
        
        # Dominant frequency
        if len(fft_vals) > 1:
            dominant_freq_idx = np.argmax(fft_vals[1:]) + 1  # Exclude DC component
            features['dominant_freq'] = float(dominant_freq_idx / len(ts))
            features['dominant_freq_power'] = float(fft_vals[dominant_freq_idx])
        else:
            features['dominant_freq'] = 0.0
            features['dominant_freq_power'] = 0.0
        
        # Power in different frequency bands
        if len(fft_vals) > 60:
            # Low frequency (cycles per day)
            features['power_low_freq'] = float(np.sum(fft_vals[1:6]))
            # Medium frequency (cycles per hour)
            features['power_med_freq'] = float(np.sum(fft_vals[6:25]))
            # High frequency
            features['power_high_freq'] = float(np.sum(fft_vals[25:]))
        else:
            features['power_low_freq'] = 0.0
            features['power_med_freq'] = 0.0
            features['power_high_freq'] = 0.0
        
        return features
    
    def _extract_activity_features(self, ts: np.ndarray) -> Dict[str, float]:
        """Extract features related to activity patterns."""
        features = {}
        
        # Active vs inactive periods
        threshold = np.mean(ts) * 0.1
        active_periods = ts > threshold
        features['active_ratio'] = float(np.sum(active_periods) / len(ts))
        
        # Burstiness
        mean_ts = np.mean(ts)
        std_ts = np.std(ts)
        if (std_ts + mean_ts) > 0:
            features['burstiness'] = float((std_ts - mean_ts) / (std_ts + mean_ts + 1e-6))
        else:
            features['burstiness'] = 0.0
        
        # Concentration of activity
        if np.sum(ts) > 0:
            normalized = ts / np.sum(ts)
            features['gini_coefficient'] = float(self._gini_coefficient(normalized))
        else:
            features['gini_coefficient'] = 0.0
        
        # Time to peak
        if np.max(ts) > 0:
            features['time_to_peak_ratio'] = float(np.argmax(ts) / len(ts))
        else:
            features['time_to_peak_ratio'] = 0.0
        
        return features
    
    def _autocorrelation(self, ts: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        if lag >= len(ts):
            return 0.0
        c0 = np.var(ts)
        if c0 == 0:
            return 0.0
        ct = np.mean((ts[:-lag] - np.mean(ts)) * (ts[lag:] - np.mean(ts)))
        return ct / c0
    
    def _spectral_entropy(self, psd: np.ndarray) -> float:
        """Compute spectral entropy of power spectral density."""
        # Remove zero values
        psd_positive = psd[psd > 0]
        if len(psd_positive) == 0:
            return 0.0
        # Normalize
        psd_norm = psd_positive / np.sum(psd_positive)
        # Compute entropy
        return -np.sum(psd_norm * np.log2(psd_norm))
    
    def _gini_coefficient(self, x: np.ndarray) -> float:
        """Compute Gini coefficient for concentration measurement."""
        sorted_x = np.sort(x)
        n = len(x)
        if n == 0:
            return 0.0
        cumsum = np.cumsum(sorted_x)
        if cumsum[-1] == 0:
            return 0.0
        return (2 * np.sum((np.arange(1, n+1) * sorted_x))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return zero features for empty time series."""
        # Create a dummy feature set with all zeros
        dummy_ts = np.ones(10)  # Small dummy series
        feature_template = self.extract_features(dummy_ts)
        return {k: 0.0 for k in feature_template.keys()}


def create_feature_matrix(time_series_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, List[str]]:
    """Create feature matrix from multiple time series.
    
    Args:
        time_series_dict: Dictionary mapping IDs to time series
        
    Returns:
        Feature DataFrame and list of function hashes
    """
    extractor = TimeSeriesFeatureExtractor()
    
    features_list = []
    function_hashes = []
    
    for func_hash, ts in time_series_dict.items():
        features = extractor.extract_features(ts)
        features_list.append(features)
        function_hashes.append(func_hash)
    
    feature_df = pd.DataFrame(features_list)
    feature_df['function_hash'] = function_hashes
    
    return feature_df, function_hashes