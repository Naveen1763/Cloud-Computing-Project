#!/usr/bin/env python3
"""
Prepare Azure Functions dataset for experiments.

This script:
1. Checks if the dataset exists
2. Provides download instructions if missing
3. Validates the dataset structure
4. Creates necessary directories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import logging
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dataset(data_dir: Path) -> bool:
    """Check if Azure Functions dataset is present."""
    required_files = []
    
    # Check for invocation files (at least day 1)
    for day in range(1, 15):
        filename = f"invocations_per_function_md.anon.d{day:02d}.csv"
        filepath = data_dir / filename
        if day == 1:  # At least day 1 must exist
            if not filepath.exists():
                # Not fatal: main() falls back to the bundled KSWD traces.
                logger.info(f"Azure file not present: {filename}")
                return False
        if filepath.exists():
            required_files.append(filename)
    
    logger.info(f"Found {len(required_files)} invocation files")
    
    # Check for duration files
    duration_files = list(data_dir.glob("function_durations_percentiles.anon.d*.csv"))
    logger.info(f"Found {len(duration_files)} duration files")
    
    # Check for memory files
    memory_files = list(data_dir.glob("app_memory_percentiles.anon.d*.csv"))
    logger.info(f"Found {len(memory_files)} memory files")
    
    return len(required_files) > 0


def validate_dataset_structure(data_dir: Path):
    """Validate the structure of the dataset."""
    logger.info("Validating dataset structure...")
    
    # Check a sample invocation file
    sample_file = data_dir / "invocations_per_function_md.anon.d01.csv"
    if sample_file.exists():
        df = pd.read_csv(sample_file, nrows=5)
        
        # Check required columns
        required_columns = ['HashOwner', 'HashApp', 'HashFunction', 'Trigger']
        time_columns = [str(i) for i in range(1, 1441)]
        
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        # Check if time columns exist
        time_cols_present = sum(1 for col in time_columns if col in df.columns)
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        if time_cols_present != 1440:
            logger.error(f"Expected 1440 time columns, found {time_cols_present}")
            return False
            
        logger.info("Dataset structure validated successfully")
        logger.info(f"Sample data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns[:10])}... (showing first 10)")
        return True
    else:
        logger.error("Cannot validate - sample file not found")
        return False


def create_directories():
    """Create necessary directories for the project."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "results",
        "results/figures",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Created {len(directories)} directories")


def main():
    """Main preparation script."""
    logger.info("Preparing AAPA simulation environment...")
    
    # Create directories
    create_directories()
    
    # Check dataset
    data_dir = Path("data/raw")
    
    if data_dir.exists() and check_dataset(data_dir):
        logger.info("Azure Functions dataset found!")

        # Validate structure
        if validate_dataset_structure(data_dir):
            logger.info("Dataset is ready for experiments!")
        else:
            logger.error("Dataset validation failed")
            return 1
    else:
        # The bundled KSWD traces are enough to run the whole pipeline, so a
        # missing Azure download is not a failure - just a smaller experiment.
        kswd_dir = Path("dataset/workloads")
        n_kswd = len(list(kswd_dir.glob("w_*.json"))) if kswd_dir.exists() else 0

        if n_kswd:
            logger.info(f"Azure dataset not present; found {n_kswd} bundled KSWD workloads.")
            logger.info("The pipeline defaults to --source kswd, so you can proceed.")
        else:
            logger.error("Neither the Azure dataset nor the bundled KSWD traces were found!")

        logger.info("To use the full Azure Functions dataset instead:")
        logger.info("1. Install Kaggle API: pip install kaggle")
        logger.info("2. Set up Kaggle credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        logger.info("3. Run: kaggle datasets download -d azurepublicdataset/azurefunctions-dataset2019 -p data/raw/")
        logger.info("4. Extract: unzip data/raw/azurefunctions-dataset2019.zip -d data/raw/")
        logger.info("Or download manually from:")
        logger.info("https://www.kaggle.com/datasets/azurepublicdataset/azurefunctions-dataset2019")

        if not n_kswd:
            return 1

    # Create a sample config file
    config_path = Path("config")
    config_path.mkdir(exist_ok=True)
    
    default_config = """# AAPA Default Configuration

# Autoscaling parameters
scaling:
  spike:
    target_cpu: 0.3
    cooldown: 1200  # 20 minutes
    warm_pool_size: 2
    min_replicas: 2
    max_replicas: 100
    
  periodic:
    target_cpu: 0.75
    cooldown: 180  # 3 minutes
    min_replicas: 1
    max_replicas: 100
    prediction_horizon: 900  # 15 minutes
    
  ramp:
    target_cpu: 0.6
    cooldown: 420  # 7 minutes
    min_replicas: 1
    max_replicas: 100
    trend_window: 10
    
  stationary_noisy:
    target_cpu: 0.55
    cooldown: 720  # 12 minutes
    min_replicas: 1
    max_replicas: 100
    stability_window: 5

# Simulation parameters
simulation:
  pod_startup_time: 30  # seconds
  metric_interval: 60  # seconds
  cpu_per_request: 0.001
  memory_per_pod: 256  # MB
  
# Classification parameters
classification:
  window_size: 60  # minutes
  stride: 10  # minutes
  min_invocations: 1000
  confidence_threshold: 0.7

# Evaluation parameters
evaluation:
  slo_target: 500  # milliseconds
  rei_weights:
    slo: 0.5
    efficiency: 0.3
    stability: 0.2
"""
    
    config_file = config_path / "default.yaml"
    with open(config_file, 'w') as f:
        f.write(default_config)
    logger.info(f"Created default configuration at {config_file}")
    
    logger.info("Preparation completed successfully!")
    logger.info("You can now run:")
    logger.info("  - python examples/run_classification.py")
    logger.info("  - python examples/run_simulation.py")
    logger.info("  - python examples/create_visualizations.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())