# Netflix Device Analytics

Netflix Device Analytics is a high scale telemetry simulation and monitoring platform designed to investigate playback performance across a global fleet of millions of devices. In a production streaming environment, identifying subtle service degradations hidden within massive datasets is a core challenge for data engineering and reliability teams. This project provides a robust framework for simulating complex failure scenarios, such as firmware rollout anomalies and regional CDN outages, and provides the necessary analytical views to detect and resolve these issues before they impact the user experience.

## Key Results

| Scenario | Affected Slices | Observed Impact | Resolution Status |
|---|---|---|---|
| Firmware Rollout Anomaly | Smart TVs, v4.2.0 | 4.2x spike in playback failures | Resolved in v4.2.1 |
| Regional CDN Failure | Asia Pacific regions | 315ms increase in median latency | Redirected to primary edge |
| Silent Metric Drift | Smart Monitors, Europe West | 95% drop in buffering reports | Re-calibrated data pipeline |

## Data Architecture

The architecture is built around a highly scalable telemetry generator that produces partitioned Parquet files resembling real world device event logs. Each event captures critical dimensions such as device type, firmware version, and geographic region alongside performance metrics like request latency and bytes transferred. This data is organized into weekly partitions to support efficient time series analysis and is validated against a machine readable manifest that defines the expected anomaly patterns for testing detection algorithms.

## Implementation

We implement the ingestion layer using a batch processing pipeline that handles millions of rows across multiple device classes. The analytical core computes rolling error rates and latency percentiles to identify statistical outliers in specific device firmware combinations. A Streamlit dashboard provides the primary interface for visualizing these telemetry streams, allowing engineers to drill down into specific regional slices and verify the impact of infrastructure changes in real time.

## Quickstart

Follow these steps to initialize the environment and generate a sample telemetry dataset for analysis.

```bash
cd netflix-device-analytics
python -m venv .venv
# On Windows PowerShell use: .\.venv\Scripts\Activate.ps1
# On Unix or Mac use: source .venv/bin/activate
pip install -r requirements.txt
# Generate 1 million events for exploration
python -m data.generator
streamlit run dashboard/app.py
```

## Reproducing Results

The results displayed in the interactive dashboard can be reproduced by executing the analytical pipeline on the generated Parquet files. The system will automatically compare the observed metric distributions against the baseline periods to highlight the simulated anomalies.
