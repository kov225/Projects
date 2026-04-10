# 📺 Streaming Intelligence: Device-Level QoS Analytics

This project implements a Quality of Service (QoS) telemetry pipeline to analyze video streaming performance across heterogeneous device categories (Smart TVs, Mobile, Web, Game Consoles). It focuses on identifying performance bottlenecks and quantifying the statistical significance of buffering latency across platforms.

## 🧠 Methodology: The Statistics of Streaming Experience

Ensuring a seamless playback experience requires monitoring low-level telemetry through a rigorous analytical lens.

### 1. QoS Metrics (Quality of Service)
- **Buffering Ratio**: Measured as $\frac{\text{Total Buffer Time}}{\text{Total Playback Time}}$. This is our primary KPI for site reliability and user churn prediction.
- **Throughput Efficiency**: Analyzing the stability of the encoding ladder based on device-specific bandwidth constraints.

### 2. Bootstrap Significance Testing
Streaming telemetry is often highly skewed (most sessions have zero buffering, while a few have extreme spikes). Standard T-tests can fail to capture the true distribution.
- **Implementation**: We use **Non-Parametric Bootstrap Resampling** to generate empirical confidence intervals for buffering differences between device categories.
- **Goal**: Formally determine if a performance lag on 'Mobile' relative to 'Smart TV' is a systemic infrastructure issue or random noise.

## 🛠️ Project Structure

```text
├── analytics/
│   ├── device_performance.py # Core QoS Engine & Bootstrap Analysis
├── ingestion/               # Telemetry ingestion scripts (WIP)
├── database/                # Schema definitions for session logging
├── Dockerfile               # Containerized analytics environment
└── docker-compose.yml       # Full stack orchestration (InfluxDB + Grafana)
```

## 🚀 Quick Start

1. **Calculate QoS Reports**:
   ```bash
   python analytics/device_performance.py
   ```

2. **Deploy Telemetry Stack**:
   ```bash
   docker compose up -d
   ```

---
*Developed as part of my Applied Data Science & ML Engineering Portfolio.*
