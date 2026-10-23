# Kubernetes Serverless Workload Dataset (KSWD)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/GuilinDev/aapa-simulator)

## Overview

The Kubernetes Serverless Workload Dataset (KSWD) is a comprehensive collection of real-world serverless workload traces, specifically formatted and enriched for Kubernetes autoscaling research. This dataset is derived from the Azure Functions 2019 public dataset and has been transformed to provide:

- **10,000+ real-world workload traces** spanning 14 days each
- **Archetype classifications** (SPIKE, PERIODIC, RAMP, STATIONARY) with confidence scores
- **Resource requirements** (CPU and memory) derived from execution characteristics
- **SLO targets** based on actual performance data
- **Rich feature sets** for machine learning applications

## Key Features

### 1. Real-World Patterns
Unlike synthetic workloads, KSWD captures the complexity and unpredictability of production serverless applications, including:
- Sudden traffic spikes
- Daily/weekly periodicity
- Gradual scaling patterns
- Noisy baseline behavior

### 2. Archetype Labels
Each workload is classified into one of four archetypes using weak supervision:
- **SPIKE (23%)**: Sudden bursts with low baseline activity
- **PERIODIC (31%)**: Regular, predictable patterns
- **RAMP (19%)**: Gradual increases or decreases
- **STATIONARY (27%)**: Stable with random noise

### 3. Kubernetes-Ready Format
All workloads include:
- Request rates (per-minute and per-second)
- CPU requirements in millicores
- Memory requirements in MB
- Response time SLO targets

## Dataset Structure

```
dataset/
├── README.md                 # This file
├── metadata.json            # Dataset metadata and statistics
├── workloads/              # Individual workload traces
│   ├── w_0001.json
│   ├── w_0002.json
│   └── ...
├── archetypes/             # Workload IDs grouped by archetype
│   ├── spike.json
│   ├── periodic.json
│   ├── ramp.json
│   └── stationary_noisy.json
├── examples/               # Usage examples
│   ├── load_workload.py
│   ├── visualize_traces.py
│   └── generate_k8s_hpa.py
└── tools/                  # Dataset generation tools
    ├── azure_to_k8s_converter.py
    └── feature_extractor.py
```

## Data Schema

Each workload JSON file contains:

```json
{
  "workload_id": "w_0001",
  "archetype": "spike",
  "archetype_confidence": 0.85,
  "duration_minutes": 20160,
  "resource_requirements": {
    "cpu_millicores": {
      "p50": 100,
      "p90": 250,
      "p99": 500
    },
    "memory_mb": {
      "p50": 128,
      "p90": 256,
      "p99": 512
    }
  },
  "slo_targets": {
    "response_time_ms": {
      "p50": 100,
      "p90": 200,
      "p99": 500
    }
  },
  "request_trace": {
    "timestamps": ["2019-07-01T00:00:00Z", ...],
    "requests_per_minute": [10, 15, 8, ...],
    "requests_per_second": [0.17, 0.25, 0.13, ...]
  },
  "features": {
    "mean_rps": 1.5,
    "variance_rps": 2.3,
    "spectral_entropy": 0.76,
    "autocorr_lag_60": 0.45,
    "peak_to_mean_ratio": 25.3
  }
}
```

## Usage Examples

### Loading a Workload

```python
import json

# Load a single workload
with open('dataset/workloads/w_0001.json', 'r') as f:
    workload = json.load(f)

# Access request trace
timestamps = workload['request_trace']['timestamps']
requests_per_minute = workload['request_trace']['requests_per_minute']

# Get resource requirements
cpu_p90 = workload['resource_requirements']['cpu_millicores']['p90']
memory_p90 = workload['resource_requirements']['memory_mb']['p90']
```

### Finding Workloads by Archetype

```python
import json

# Load archetype mappings
with open('dataset/archetypes/spike.json', 'r') as f:
    spike_workloads = json.load(f)

# Load all spike workloads
for workload_id in spike_workloads['workload_ids']:
    with open(f'dataset/workloads/{workload_id}.json', 'r') as f:
        workload = json.load(f)
        # Process spike workload
```

### Generating Kubernetes HPA Configuration

```python
# See examples/generate_k8s_hpa.py for full implementation
from examples.generate_k8s_hpa import generate_hpa_config

workload = load_workload('w_0001')
hpa_yaml = generate_hpa_config(
    workload,
    target_utilization=0.7,
    min_replicas=1,
    max_replicas=100
)
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Workloads | 10,148 |
| Duration per Workload | 14 days |
| Time Granularity | 1 minute |
| Total Data Points | 144M+ |
| Archetype Distribution | SPIKE: 23%, PERIODIC: 31%, RAMP: 19%, STATIONARY: 27% |

## Resource Estimation Methodology

CPU requirements are estimated using:
```
cpu_millicores = (avg_execution_time_ms * requests_per_second * 1000) / target_utilization
```

Memory requirements are derived from the Azure Functions memory allocation data with a 20% buffer for Kubernetes overhead.

## Limitations

1. **Temporal Resolution**: Original data is aggregated at 1-minute intervals
2. **Resource Estimates**: CPU/memory are derived, not directly measured
3. **Time Period**: Data from July 2019 may not reflect current patterns
4. **Function Types**: Primarily represents event-driven serverless workloads

## Citation

If you use this dataset in your research, please cite both:

1. Our paper:
```bibtex
@inproceedings{zhang2025aapa,
  title={Archetype-Aware Predictive Autoscaling with Uncertainty Quantification for Serverless Workloads on Kubernetes},
  author={Zhang, Guilin and others},
  booktitle={IEEE High Performance Extreme Computing Conference (HPEC)},
  year={2025}
}
```

2. The original Azure dataset:
```bibtex
@inproceedings{shahrad2020serverless,
  title={Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider},
  author={Shahrad, Mohammad and others},
  booktitle={USENIX Annual Technical Conference},
  year={2020}
}
```

## License

This dataset is released under the MIT License. See the LICENSE file in the parent directory for details.

## Contributing

We welcome contributions to improve the dataset! Please see:
- Report issues: https://github.com/GuilinDev/aapa-simulator/issues
- Dataset generation code: `tools/azure_to_k8s_converter.py`

## Acknowledgments

- Microsoft Azure team for releasing the original Functions dataset
- IEEE HPEC community for valuable feedback
- All researchers who have used and provided feedback on this dataset