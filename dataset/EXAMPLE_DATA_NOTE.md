# Example Dataset Note

This directory contains **example workload files** for demonstration purposes. Please note:

## Current Status

- **5 example workloads** (w_0001 to w_0005) representing all 4 archetypes
- **Truncated traces**: Each file shows only the first 100 minutes of data for readability
- **Full duration**: Each workload actually represents 14 days (20,160 minutes) of data
- **All 4 archetypes** are represented:
  - SPIKE: w_0001, w_0003
  - PERIODIC: w_0002
  - RAMP: w_0004
  - STATIONARY_NOISY: w_0005

## Full Dataset Generation

To generate the complete dataset with full 14-day traces from Azure Functions data:

```bash
# Assuming you have downloaded the Azure Functions 2019 dataset
python dataset/tools/azure_to_k8s_converter.py \
  --azure-path /path/to/azure-functions-2019/ \
  --output-path dataset/ \
  --max-workloads 10000
```

This will create:
- 10,000+ workload files with complete 14-day traces
- Accurate feature extraction using scipy
- Real-world patterns from Azure Functions

## Why Example Data?

The example files serve to:
1. Demonstrate the data format and schema
2. Allow testing of the example scripts without downloading the full Azure dataset
3. Provide quick visualization of different archetype patterns
4. Enable rapid prototyping and development

## File Size Considerations

- Full workload files with 20,160 data points each would be ~1-2MB per file
- The complete dataset would be several GB
- Example files are kept small (~15KB) for repository efficiency

## Next Steps

1. Download the Azure Functions 2019 dataset from Kaggle
2. Run the conversion script to generate the full dataset
3. Use the generated dataset for your research and experiments

For questions or issues, please see the main repository documentation.