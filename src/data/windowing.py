import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from src.data.features import TimeSeriesFeatureExtractor

logger = logging.getLogger(__name__)

class SlidingWindowProcessor:
    """Process time series data using sliding windows for feature extraction and labeling."""
    
    def __init__(self, window_size: int = 60, stride: int = 10):
        """Initialize sliding window processor.
        
        Args:
            window_size: Size of the window in minutes (default: 60)
            stride: Stride/step size in minutes (default: 10)
        """
        self.window_size = window_size
        self.stride = stride
        self.feature_extractor = TimeSeriesFeatureExtractor()
        
    def create_sliding_windows(self, time_series: np.ndarray) -> List[np.ndarray]:
        """Create sliding windows from a time series.
        
        Args:
            time_series: Full time series data
            
        Returns:
            List of window arrays
        """
        windows = []
        
        # Calculate number of windows
        n_windows = (len(time_series) - self.window_size) // self.stride + 1
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            if end_idx <= len(time_series):
                window = time_series[start_idx:end_idx]
                windows.append(window)
                
        return windows
    
    def process_function_time_series(self, func_hash: str, time_series: np.ndarray) -> pd.DataFrame:
        """Process a single function's time series into windowed features.
        
        Args:
            func_hash: Function identifier
            time_series: Complete time series for the function
            
        Returns:
            DataFrame with features for each window
        """
        windows = self.create_sliding_windows(time_series)
        
        if len(windows) == 0:
            return pd.DataFrame()
        
        features_list = []
        
        for window_idx, window in enumerate(windows):
            # Extract features for this window
            features = self.feature_extractor.extract_features(window)
            
            # Add metadata
            features['function_hash'] = func_hash
            features['window_idx'] = window_idx
            features['window_start_minute'] = window_idx * self.stride
            features['window_end_minute'] = window_idx * self.stride + self.window_size
            features['day'] = features['window_start_minute'] // 1440
            features['hour_of_day'] = (features['window_start_minute'] % 1440) // 60
            
            features_list.append(features)
            
        return pd.DataFrame(features_list)
    
    def process_all_functions(self, time_series_dict: Dict[str, np.ndarray], 
                            filter_http_only: bool = True,
                            min_activity_threshold: float = 0.1) -> pd.DataFrame:
        """Process all functions into windowed features.
        
        Args:
            time_series_dict: Dictionary mapping function hashes to time series
            filter_http_only: Whether to filter only HTTP functions (if metadata available)
            min_activity_threshold: Minimum activity ratio to include a function
            
        Returns:
            DataFrame with all windowed features
        """
        all_features = []
        
        logger.info(f"Processing {len(time_series_dict)} functions with window_size={self.window_size}, stride={self.stride}")
        
        for func_hash, time_series in tqdm(time_series_dict.items(), desc="Processing functions"):
            # Check activity level
            activity_ratio = np.sum(time_series > 0) / len(time_series)
            if activity_ratio < min_activity_threshold:
                continue
                
            # Process this function
            func_features = self.process_function_time_series(func_hash, time_series)
            
            if len(func_features) > 0:
                all_features.append(func_features)
                
        if all_features:
            result_df = pd.concat(all_features, ignore_index=True)
            logger.info(f"Created {len(result_df)} window samples from {len(all_features)} functions")
            return result_df
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()
    
    def get_window_statistics(self, features_df: pd.DataFrame) -> Dict:
        """Get statistics about the windowed data.
        
        Args:
            features_df: DataFrame with windowed features
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_windows': len(features_df),
            'unique_functions': features_df['function_hash'].nunique(),
            'windows_per_function': len(features_df) / features_df['function_hash'].nunique(),
            'days_covered': features_df['day'].max() + 1 if len(features_df) > 0 else 0,
            'feature_columns': [col for col in features_df.columns if col not in 
                              ['function_hash', 'window_idx', 'window_start_minute', 
                               'window_end_minute', 'day', 'hour_of_day']]
        }
        
        return stats


def create_train_test_split(windowed_features_df: pd.DataFrame, 
                           test_days: List[int] = [12, 13], 
                           val_days: List[int] = [10, 11]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split windowed features into train/val/test sets by day.
    
    Args:
        windowed_features_df: DataFrame with windowed features
        test_days: Days to use for test set
        val_days: Days to use for validation set
        
    Returns:
        train_df, val_df, test_df
    """
    test_df = windowed_features_df[windowed_features_df['day'].isin(test_days)]
    val_df = windowed_features_df[windowed_features_df['day'].isin(val_days)]
    train_df = windowed_features_df[~windowed_features_df['day'].isin(test_days + val_days)]
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing sliding window processor...")
    
    # Create synthetic 3-day time series
    test_series = {
        'func1': np.random.poisson(10, 1440 * 3),  # 3 days
        'func2': np.random.poisson(50, 1440 * 3)
    }
    
    # Process with sliding windows
    processor = SlidingWindowProcessor(window_size=60, stride=10)
    features_df = processor.process_all_functions(test_series)
    
    print(f"\nExtracted {len(features_df)} windows")
    print(f"Columns: {features_df.columns.tolist()}")
    
    # Get statistics
    stats = processor.get_window_statistics(features_df)
    print(f"\nStatistics:")
    for key, value in stats.items():
        if key != 'feature_columns':
            print(f"  {key}: {value}")
    
    # Test train/test split
    train_df, val_df, test_df = create_train_test_split(features_df, test_days=[2], val_days=[1])
    print(f"\nSplit results:")
    print(f"  Train: {len(train_df)} windows")
    print(f"  Val: {len(val_df)} windows")
    print(f"  Test: {len(test_df)} windows")

def create_workload_split(windowed_features_df: pd.DataFrame,
                          label_col: str = 'label',
                          test_frac: float = 0.25,
                          val_frac: float = 0.25,
                          seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by workload so no workload appears in more than one split.

    ``create_train_test_split`` divides by day, which puts days 1-9 of a
    workload in training and days 12-13 of the *same* workload in test. Since
    the archetype label is constant per workload, a model can score highly by
    recognising the workload and recalling its label - it never has to learn
    what an archetype looks like. That inflates accuracy and collapses
    predicted probabilities toward 1, which in turn makes conformal prediction
    sets degenerate to singletons.

    Splitting by workload forces genuine generalisation to unseen traces.
    Workloads are drawn per archetype so every class appears in every split,
    which matters because SPIKE is rare and would otherwise land entirely on
    one side.

    Args:
        windowed_features_df: Windowed features, one row per window.
        label_col: Column holding the archetype label.
        test_frac: Fraction of each archetype's workloads held out for test.
        val_frac: Fraction held out for validation.
        seed: Seed for the workload shuffle.

    Returns:
        train_df, val_df, test_df - disjoint in ``function_hash``.
    """
    rng = np.random.default_rng(seed)

    workload_labels = (
        windowed_features_df.groupby('function_hash')[label_col].agg(
            lambda s: s.value_counts().index[0]
        )
    )

    train_ids, val_ids, test_ids = [], [], []
    for archetype in sorted(workload_labels.unique()):
        ids = np.sort(workload_labels.index[workload_labels == archetype].to_numpy())
        ids = rng.permutation(ids)
        n = len(ids)

        # With very few workloads of a class, guarantee at least one in each
        # split before proportional allocation.
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - n_test - 1))

        test_ids.extend(ids[:n_test])
        val_ids.extend(ids[n_test:n_test + n_val])
        train_ids.extend(ids[n_test + n_val:])

    def subset(ids):
        return windowed_features_df[windowed_features_df['function_hash'].isin(set(ids))]

    train_df, val_df, test_df = subset(train_ids), subset(val_ids), subset(test_ids)

    logger.info(
        f"Workload split - train: {len(train_df)} windows / {len(train_ids)} workloads, "
        f"val: {len(val_df)} / {len(val_ids)}, test: {len(test_df)} / {len(test_ids)}"
    )
    overlap = (set(train_ids) & set(test_ids)) | (set(train_ids) & set(val_ids))
    if overlap:
        raise AssertionError(f"workload leaked across splits: {overlap}")

    return train_df, val_df, test_df
