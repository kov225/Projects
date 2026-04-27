# Streaming QoS Analytics

**Stack:** Python, NumPy, SciPy (bootstrap), Pandas, Docker Compose, Kafka,
Grafana.

A streaming Quality of Service (QoS) study comparing buffering and
throughput across device classes (Smart TV, mobile, web, game console). The
current codebase is the **analytics layer** of a larger telemetry pipeline.
The ingestion to store path is scaffolded but not yet wired end to end.

## What the analytics module does

### Buffering ratio
The primary KPI is the per session buffering ratio:

```
buffering_ratio = total_buffer_time / total_playback_time
```

This collapses noisy low level metrics into a single, churn relevant signal.

### Bootstrap significance for buffering differences
Buffering distributions are heavy tailed and zero inflated: most sessions
have no rebuffer events, but a small fraction have very long ones. A
parametric t-test handles this badly. The script uses a non parametric
bootstrap (1,000 resamples by default) of the difference in mean buffering
ratio between two device classes to produce a 95 percent confidence
interval for the gap, which holds up under skew.

### Throughput stability
A secondary view fits the variance of measured throughput within an ABR
"ladder" rung, which is a proxy for whether the bitrate adapter is settling
into a stable rung or oscillating.

## Repository layout

```
analytics/
  device_performance.py    QoSAnalyzer + bootstrap_buffering_diff
ingestion/                 Producer scaffolding (work in progress)
database/                  Schema sketches for session logging
dashboard/                 Grafana provisioning (work in progress)
docker-compose.yml         Kafka, Zookeeper, Grafana
Dockerfile
```

## Reproduction

The analytics script runs against a synthetic session generator:

```bash
pip install -r requirements.txt
python analytics/device_performance.py
```

The Compose file brings up Kafka and Grafana for development. Note that
Grafana is provisioned but the dashboard JSON is not yet pinned. That is
part of the unfinished ingestion path.

## Honest scope

This project is currently a **proof of concept analytics layer** rather
than a working telemetry system. To be specific about what is and is not
done:

- **Implemented:** the QoS metrics, the bootstrap CI for between device
  comparisons, the synthetic session generator that drives the analysis.
- **Not yet implemented:** the Kafka consumer that writes real sessions to
  the store; an InfluxDB or TimescaleDB sink; the Grafana dashboard
  definitions; integration tests across the pipeline.
- **Earlier README claim that needs correcting:** the README originally
  mentioned an "InfluxDB plus Grafana" stack; the Compose file does not
  currently include InfluxDB. Choosing between InfluxDB, TimescaleDB, and
  ClickHouse for the time series sink is the open design question.

The analytics layer is intentionally separable so it can be reused once the
ingestion path is finished.
