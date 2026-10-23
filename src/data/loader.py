import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureFunctionsDataLoader:
    """Load and process Azure Functions dataset for workload analysis."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.invocation_data = {}
        self.duration_data = {}
        self.memory_data = {}
        
    def load_invocation_data(self, days: List[int] = None) -> Dict[str, np.ndarray]:
        """Load invocation data for specified days.
        
        Args:
            days: List of day numbers (1-14). If None, loads all days.
            
        Returns:
            Dictionary mapping function hash to invocation time series.
        """
        if days is None:
            days = list(range(1, 15))
            
        logger.info(f"Loading invocation data for days: {days}")
        
        all_functions = {}
        
        for day in days:
            file_path = self.data_dir / f"invocations_per_function_md.anon.d{day:02d}.csv"
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            # Extract time series columns (1-1440)
            time_columns = [str(i) for i in range(1, 1441)]
            
            for _, row in df.iterrows():
                func_hash = row['HashFunction']
                time_series = row[time_columns].values.astype(float)
                
                if func_hash not in all_functions:
                    all_functions[func_hash] = []
                all_functions[func_hash].append(time_series)
                
        # Concatenate daily time series for each function
        for func_hash in all_functions:
            all_functions[func_hash] = np.concatenate(all_functions[func_hash])
            
        logger.info(f"Loaded {len(all_functions)} unique functions")
        self.invocation_data = all_functions
        return all_functions
    
    def load_duration_data(self, days: List[int] = None) -> pd.DataFrame:
        """Load function duration statistics."""
        if days is None:
            days = list(range(1, 15))
            
        logger.info(f"Loading duration data for days: {days}")
        
        duration_dfs = []
        for day in days:
            file_path = self.data_dir / f"function_durations_percentiles.anon.d{day:02d}.csv"
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path)
            duration_dfs.append(df)
            
        if duration_dfs:
            self.duration_data = pd.concat(duration_dfs, ignore_index=True)
            # Group by function and average the statistics
            self.duration_data = self.duration_data.groupby('HashFunction').agg({
                'Average': 'mean',
                'Count': 'sum',
                'Minimum': 'min',
                'Maximum': 'max',
                'percentile_Average_50': 'mean',
                'percentile_Average_99': 'mean'
            }).reset_index()
        else:
            self.duration_data = pd.DataFrame()
            
        return self.duration_data
    
    def load_memory_data(self, days: List[int] = None) -> pd.DataFrame:
        """Load application memory statistics."""
        if days is None:
            days = list(range(1, 13))  # Only 12 days of memory data
            
        logger.info(f"Loading memory data for days: {days}")
        
        memory_dfs = []
        for day in days:
            file_path = self.data_dir / f"app_memory_percentiles.anon.d{day:02d}.csv"
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path)
            memory_dfs.append(df)
            
        if memory_dfs:
            self.memory_data = pd.concat(memory_dfs, ignore_index=True)
            # Group by app and average the statistics
            self.memory_data = self.memory_data.groupby('HashApp').agg({
                'AverageAllocatedMb': 'mean',
                'AverageAllocatedMb_pct50': 'mean',
                'AverageAllocatedMb_pct99': 'mean'
            }).reset_index()
        else:
            self.memory_data = pd.DataFrame()
            
        return self.memory_data
    
    def filter_functions(self, min_invocations: int = 100) -> Dict[str, np.ndarray]:
        """Filter functions by minimum total invocations.
        
        Args:
            min_invocations: Minimum total invocations required.
            
        Returns:
            Filtered dictionary of function time series.
        """
        filtered = {}
        for func_hash, time_series in self.invocation_data.items():
            total_invocations = np.sum(time_series)
            if total_invocations >= min_invocations:
                filtered[func_hash] = time_series
                
        logger.info(f"Filtered to {len(filtered)} functions with >= {min_invocations} invocations")
        return filtered
    
    def get_function_metadata(self, func_hash: str) -> Dict:
        """Get duration and memory metadata for a function."""
        metadata = {'hash': func_hash}
        
        # Get duration info
        if not self.duration_data.empty and func_hash in self.duration_data['HashFunction'].values:
            duration_row = self.duration_data[self.duration_data['HashFunction'] == func_hash].iloc[0]
            metadata['avg_duration_ms'] = duration_row['Average']
            metadata['p50_duration_ms'] = duration_row['percentile_Average_50']
            metadata['p99_duration_ms'] = duration_row['percentile_Average_99']
        else:
            metadata['avg_duration_ms'] = 100  # Default 100ms
            metadata['p50_duration_ms'] = 50
            metadata['p99_duration_ms'] = 500
            
        # Note: Memory is per app, not per function
        # Would need app-function mapping from invocation data
        metadata['avg_memory_mb'] = 256  # Default 256MB
        
        return metadata


if __name__ == "__main__":
    # Test the data loader
    loader = AzureFunctionsDataLoader("/home/guilin/Downloads/azurefunctions-dataset2019")
    
    # Load first 3 days as a test
    functions = loader.load_invocation_data(days=[1, 2, 3])
    duration_df = loader.load_duration_data(days=[1, 2, 3])
    memory_df = loader.load_memory_data(days=[1, 2, 3])
    
    print(f"Loaded {len(functions)} functions")
    print(f"Duration data shape: {duration_df.shape}")
    print(f"Memory data shape: {memory_df.shape}")
    
    # Filter and show sample
    filtered = loader.filter_functions(min_invocations=1000)
    if filtered:
        sample_hash = list(filtered.keys())[0]
        print(f"\nSample function {sample_hash}:")
        print(f"Time series length: {len(filtered[sample_hash])}")
        print(f"Total invocations: {np.sum(filtered[sample_hash])}")
        print(f"Metadata: {loader.get_function_metadata(sample_hash)}")