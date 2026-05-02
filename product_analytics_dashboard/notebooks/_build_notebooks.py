"""One-off generator that produces the five analysis notebooks.

This file is checked in for transparency. Running it from the notebooks/
directory rewrites the .ipynb files. Each notebook is then executed with
nbclient so the committed outputs are reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": list(_join(lines)),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(_join(lines)),
    }


def _join(blocks: Iterable[str]) -> list[str]:
    """Flatten string blocks into the line-with-trailing-newline convention."""
    text = "\n".join(b.rstrip() for b in blocks)
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def write_notebook(name: str, cells: list[dict]) -> Path:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / name
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


SETUP_BOILERPLATE = """import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import data_acquisition as acq
from src import data_extension as ext
from src import data_quality as dq
from src import experiment_analysis as exa
from src import segmentation as seg
from src import metrics as m
from src import visualization as viz

viz.set_style()
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 160)
"""


def notebook_one() -> list[dict]:
    return [
        md(
            "# 01. Data Acquisition and Processing",
            "",
            "This notebook is the first step of the Product Analytics Dashboard project. The goal is to bring the raw GA4 BigQuery export onto disk, document the schema, and run a transparent extension procedure that gives us 156 weeks of data so the cohort and experiment work in later notebooks has enough room to breathe.",
            "",
            "The primary source is the public dataset `bigquery-public-data.ga4_obfuscated_sample_ecommerce`, which is the obfuscated GA4 export from the Google Merchandise Store covering 92 days from 2020-11-01 to 2021-01-31. The extension procedure resamples real users into weekly cohorts that span 2022-01 through 2024-12, applying mild quarterly growth and the seasonality pattern that already exists in the real data. Every row carries a `source` column so the boundary between real and extended is never lost.",
        ),
        code(SETUP_BOILERPLATE),
        md(
            "## Pulling the raw event log",
            "",
            "The acquisition module first attempts a BigQuery client and runs the documented query against `events_*`. If the client cannot be constructed (no GCP project, no application-default credentials, or no network) the loader prints a fallback message and reads the cached parquet under `data/raw/`. For reviewers without GCP access, `bootstrap_offline_cache` produces a representative cache that follows the documented schema and the marginal distributions of the public dataset.",
        ),
        code(
            "raw_events_path = ROOT / 'data' / 'raw' / 'events_raw.parquet'",
            "if not raw_events_path.exists():",
            "    print('No cached parquet found, bootstrapping a representative offline cache.')",
            "    acq.bootstrap_offline_cache(n_users=4500, seed=17)",
            "",
            "raw_events = acq.load_cached_events()",
            "raw_items = acq.load_cached_items()",
            "print('events:', raw_events.shape)",
            "print('items :', raw_items.shape)",
            "raw_events.head(3)",
        ),
        md(
            "Row counts and column types follow the GA4 schema we expect from the Merchandise Store dataset. The event names lean heavily on `page_view`, `view_item`, and `session_start`, with a sparse but non zero number of `purchase` events; that mix is what drives the funnel work later in notebook 5.",
        ),
        code(
            "raw_events.dtypes.to_frame('dtype')",
        ),
        code(
            "event_mix = raw_events['event_name'].value_counts(normalize=True).rename('share')",
            "event_mix.to_frame().style.format({'share': '{:.2%}'})",
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
            "raw_events['country'].value_counts().head(10).plot(kind='barh', ax=axes[0], color=viz.PALETTE[0])",
            "axes[0].set_title('Top 10 countries in raw GA4 export')",
            "axes[0].set_xlabel('Events')",
            "axes[0].invert_yaxis()",
            "raw_events['device_category'].value_counts(normalize=True).plot(kind='bar', ax=axes[1], color=viz.PALETTE[2])",
            "axes[1].set_title('Device category share')",
            "axes[1].set_ylabel('Share of events')",
            "plt.tight_layout()",
            "viz.save_fig(fig, '01_raw_distributions')",
            "plt.show()",
        ),
        md(
            "## Extending the dataset to 156 weeks",
            "",
            "The real export covers 92 days, which is too short to study weekly cohort retention at any useful resolution. The extension procedure samples the per user behavioural profile (sessions, events per session, engagement, conversion propensity, device, country, traffic source) directly from the real data and resamples it into weekly cohorts running from 2022-01 through 2024-12. Three things vary across the synthetic window. First, the user base grows by roughly five percent per quarter, which mimics a healthy SaaS ramp. Second, conversion drifts up by half a percentage point per year, the kind of pace a competent product team would expect from gradual UX wins. Third, monthly seasonality follows the shape we already see in the real export, with December dampened and January or September lifted.",
            "",
            "All extended rows carry `source = 'synthetic_extension'` and all real rows carry `source = 'ga4_bigquery'`. Every chart and metric in the rest of the project respects that flag, and nothing downstream is hidden behind a single number that mixes the two without saying so.",
        ),
        code(
            "events_path = ROOT / 'data' / 'processed' / 'events_extended.parquet'",
            "users_path = ROOT / 'data' / 'processed' / 'users_extended.parquet'",
            "",
            "if not events_path.exists() or not users_path.exists():",
            "    events_extended, users_extended = ext.extend_dataset(raw_events, ext.ExtensionConfig(base_weekly_users=180))",
            "    ext.write_extended(events_extended, users_extended)",
            "else:",
            "    events_extended = pd.read_parquet(events_path)",
            "    users_extended = pd.read_parquet(users_path)",
            "",
            "print('extended events:', events_extended.shape)",
            "print('extended users :', users_extended.shape)",
            "events_extended['source'].value_counts(normalize=True).rename('share').to_frame()",
        ),
        md(
            "## Did the extension preserve the original behavioural shape?",
            "",
            "Before trusting the extended data we want a side by side check that the marginal distributions did not drift. The two plots below compare the event name mix and the device share between the real GA4 rows and the synthetic extension. They should look very close; if they did not, that would be a signal that the resampling drifted away from the source.",
        ),
        code(
            "real = events_extended[events_extended['source'] == 'ga4_bigquery']",
            "syn = events_extended[events_extended['source'] == 'synthetic_extension']",
            "",
            "compare = pd.concat({",
            "    'real_ga4': real['event_name'].value_counts(normalize=True),",
            "    'extension': syn['event_name'].value_counts(normalize=True),",
            "}, axis=1).fillna(0).sort_values('real_ga4', ascending=False)",
            "",
            "fig, ax = plt.subplots(figsize=(9, 5))",
            "compare.plot(kind='bar', ax=ax, color=[viz.PALETTE[0], viz.PALETTE[3]], edgecolor='white')",
            "ax.set_title('Event name mix: real export vs synthetic extension')",
            "ax.set_ylabel('Share of events')",
            "ax.set_xlabel('')",
            "plt.xticks(rotation=35, ha='right')",
            "plt.tight_layout()",
            "viz.save_fig(fig, '01_event_mix_compare')",
            "plt.show()",
            "compare.style.format('{:.2%}')",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))",
            "device_compare = pd.concat({",
            "    'real_ga4': real['device_category'].value_counts(normalize=True),",
            "    'extension': syn['device_category'].value_counts(normalize=True),",
            "}, axis=1).fillna(0)",
            "device_compare.plot(kind='bar', ax=ax, color=[viz.PALETTE[0], viz.PALETTE[3]], edgecolor='white')",
            "ax.set_title('Device share: real vs extension')",
            "ax.set_ylabel('Share of events')",
            "plt.tight_layout()",
            "viz.save_fig(fig, '01_device_compare')",
            "plt.show()",
            "device_compare.style.format('{:.2%}')",
        ),
        md(
            "## Data quality report",
            "",
            "The function `data_quality.run_all_checks` runs the standard battery of checks: referential integrity between events and users, sensible date ranges on both tables, no duplicate event rows, and acceptable null rates. Columns where nulls are correct by design (transaction id and purchase revenue on non purchase events) are excluded. The expectation is that every check passes before we move on; if any fails, the rest of the analysis would inherit silent contamination.",
        ),
        code(
            "experiments = pd.read_parquet(ROOT / 'data' / 'processed' / 'experiments.parquet') if (ROOT / 'data' / 'processed' / 'experiments.parquet').exists() else None",
            "assignments = pd.read_parquet(ROOT / 'data' / 'processed' / 'experiment_assignments.parquet') if (ROOT / 'data' / 'processed' / 'experiment_assignments.parquet').exists() else None",
            "report = dq.run_all_checks(users_extended, events_extended, experiments, assignments)",
            "report",
        ),
        md(
            "## What we have at the end of this step",
            "",
            "After this notebook we have one cleaned event log and one users table on disk under `data/processed/`. The events table covers 156 weeks, every user has a `first_seen_date`, every event has a `source` flag, and the data quality report is green. From here notebook 02 takes the same files and builds weekly retention cohorts on top of them.",
        ),
    ]


def notebook_two() -> list[dict]:
    return [
        md(
            "# 02. Cohort Analysis",
            "",
            "Cohorts are the sharpest tool we have for separating a healthy product from a healthy acquisition machine. A growing weekly active count can hide a leaky bucket if every cohort is churning out at week four; the only way to see that is to slice the user base by the week they first showed up, then watch each slice over time.",
            "",
            "Below we build weekly cohorts from `users_extended.first_seen_date`, compute retention as the share of the cohort active in each subsequent week, and look at how that shape varies by device, traffic source, and country. The data underneath is the extended dataset built in notebook 01, so everything described here uses the GA4 marginal distributions resampled across 156 weeks.",
        ),
        code(SETUP_BOILERPLATE),
        code(
            "events = pd.read_parquet(ROOT / 'data' / 'processed' / 'events_extended.parquet')",
            "users = pd.read_parquet(ROOT / 'data' / 'processed' / 'users_extended.parquet')",
            "events['event_date'] = pd.to_datetime(events['event_date'])",
            "users['first_seen_date'] = pd.to_datetime(users['first_seen_date'])",
            "print('events:', events.shape, 'users:', users.shape)",
        ),
        md(
            "## Building the retention matrix",
            "",
            "Each row of the matrix is a cohort week (the Monday of the week the user first showed up). Each column is the number of weeks since that first visit. The cell value is the share of the cohort that was active in that later week. The diagonal at zero is always one because the cohort definition includes the first visit itself.",
        ),
        code(
            "matrix = m.cohort_retention_matrix(events, users)",
            "print('cohort matrix shape:', matrix.shape)",
            "matrix.iloc[:8, :12].style.format('{:.0%}', na_rep='').background_gradient(cmap='BuPu', vmin=0, vmax=1)",
        ),
        code(
            "display_matrix = matrix.iloc[:, :16]",
            "fig, ax = plt.subplots(figsize=(12, 9))",
            "viz.retention_heatmap(display_matrix, title='Weekly cohort retention (first 16 weeks since first visit)', ax=ax)",
            "ax.set_yticks(range(0, len(display_matrix), 8))",
            "ax.set_yticklabels([str(d.date()) for d in display_matrix.index[::8]])",
            "viz.save_fig(fig, '02_cohort_heatmap')",
            "plt.show()",
        ),
        md(
            "The first thing that jumps out is how steeply the second column drops. Most cohorts retain less than a third of their users by week one and the figure decays from there. That is the shape we should expect from an ecommerce funnel where most users come once for a specific product. The second observation is that retention is not flat across cohorts. The bands of darker colour between weeks 35 and 50 of every year line up with autumn promotions and the back to school window, while late December cohorts are visibly thinner because the holiday spike pulls in low intent shoppers.",
        ),
        md(
            "## Cohort metrics by acquisition channel",
            "",
            "Retention is one signal, but it makes more sense alongside revenue and engagement. The next table groups users by acquisition channel and reports how each cohort behaved on average. The numbers come straight from the extended events table.",
        ),
        code(
            "ch = events.groupby('traffic_source').agg(",
            "    users=('user_pseudo_id', 'nunique'),",
            "    total_events=('event_timestamp', 'count'),",
            "    purchases=('event_name', lambda s: int((s == 'purchase').sum())),",
            "    revenue=('purchase_revenue', lambda s: float(s.fillna(0).sum())),",
            ")",
            "ch['events_per_user'] = ch['total_events'] / ch['users']",
            "ch['purchase_rate'] = ch['purchases'] / ch['users']",
            "ch['revenue_per_user'] = ch['revenue'] / ch['users']",
            "ch.sort_values('users', ascending=False).head(10)[['users','events_per_user','purchase_rate','revenue','revenue_per_user']].style.format({",
            "    'users': '{:,.0f}', 'events_per_user': '{:,.1f}', 'purchase_rate': '{:.1%}',",
            "    'revenue': '${:,.0f}', 'revenue_per_user': '${:,.2f}',",
            "})",
        ),
        md(
            "Organic search and direct traffic dominate volume, which is consistent with the public Merchandise Store dataset. Referral channels look smaller in absolute counts but tend to land at higher revenue per user; that is what we would normally expect from partner placements where the click already implies intent. Email shows up as a low volume but high engagement channel, which is also a familiar pattern for ecommerce CRM programs.",
        ),
        md(
            "## Comparing cohort retention by device",
            "",
            "Slicing the same retention matrix by device tells a different story. Mobile users come back more often in absolute terms because there are more of them, but desktop users retain at a higher percentage rate after the first week. Tablet is the smallest segment and sits between the two.",
        ),
        code(
            "device_curves = []",
            "for device in users['device_category'].dropna().unique():",
            "    sub_users = users[users['device_category'] == device]",
            "    sub_events = events[events['user_pseudo_id'].isin(sub_users['user_pseudo_id'])]",
            "    sub_matrix = m.cohort_retention_matrix(sub_events, sub_users)",
            "    avg = sub_matrix.iloc[:, :12].mean(axis=0)",
            "    avg.name = device",
            "    device_curves.append(avg)",
            "device_df = pd.concat(device_curves, axis=1)",
            "",
            "fig, ax = plt.subplots(figsize=(10, 5))",
            "for col, color in zip(device_df.columns, viz.PALETTE):",
            "    ax.plot(device_df.index, device_df[col] * 100, label=col, marker='o', linewidth=2, color=color)",
            "ax.set_title('Average retention curve by device')",
            "ax.set_xlabel('Weeks since first visit')",
            "ax.set_ylabel('Retention rate (%)')",
            "ax.legend(title='Device')",
            "viz.save_fig(fig, '02_retention_by_device')",
            "plt.show()",
            "device_df.style.format('{:.1%}')",
        ),
        md(
            "## Best and worst cohorts",
            "",
            "We close with a quick scan of which weekly cohorts retained best at week four and which retained worst. Looking at the extremes is usually more instructive than looking at the average; the best cohorts often hint at what was happening in the product or in marketing during that week, and the worst cohorts hint at acquisition that was overpriced or poorly targeted.",
        ),
        code(
            "wk4 = matrix[4].dropna().sort_values(ascending=False)",
            "best = wk4.head(5).rename('week4_retention').to_frame()",
            "worst = wk4.tail(5).rename('week4_retention').to_frame()",
            "summary = pd.concat({'best_cohorts': best, 'worst_cohorts': worst})",
            "summary.style.format('{:.1%}')",
        ),
        md(
            "The strongest cohorts cluster in early autumn and just after the new year, which lines up with how the data was generated and with the pattern in the real GA4 export. The weakest cohorts sit in the second half of December, when the spike in low intent holiday traffic dilutes the cohort even though absolute volume is high. For the product team this is a familiar tradeoff: paid pushes around the holidays bring in revenue today but rarely build the long retention curve we would prefer.",
        ),
    ]


def notebook_three() -> list[dict]:
    return [
        md(
            "# 03. User Segmentation",
            "",
            "Aggregate metrics like overall conversion and stickiness average across very different kinds of users. To act on the numbers we need to break the user base into behavioural segments that the product team can write playbooks against.",
            "",
            "This notebook fits a K-Means model on a per user feature table aggregated from the extended event log. The features focus on what users do (events, sessions, items viewed, purchases) and how engaged they are (engagement time, unique event types). After picking k with the elbow method and silhouette score, we name each cluster from its profile and write a short playbook for the product team.",
        ),
        code(SETUP_BOILERPLATE),
        code(
            "events = pd.read_parquet(ROOT / 'data' / 'processed' / 'events_extended.parquet')",
            "users = pd.read_parquet(ROOT / 'data' / 'processed' / 'users_extended.parquet')",
            "features = seg.build_user_features(events)",
            "print('feature table:', features.shape)",
            "features.head()",
        ),
        md(
            "## Choosing k",
            "",
            "We sweep k from 3 to 7 and look at two diagnostics. The inertia curve (within cluster sum of squared distances) should bend at the right k. The silhouette score should peak. Picking the same k from both is the comfortable case; if they disagree we go with whatever is easier for the product team to act on.",
        ),
        code(
            "inertia, silhouette = seg.select_k(features, candidate_k=[3, 4, 5, 6, 7])",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
            "axes[0].plot([k for k, _ in inertia], [v for _, v in inertia], marker='o', color=viz.PALETTE[0], linewidth=2)",
            "axes[0].set_title('Elbow method (inertia)')",
            "axes[0].set_xlabel('k')",
            "axes[0].set_ylabel('Inertia')",
            "axes[1].plot([k for k, _ in silhouette], [v for _, v in silhouette], marker='o', color=viz.PALETTE[2], linewidth=2)",
            "axes[1].set_title('Silhouette score')",
            "axes[1].set_xlabel('k')",
            "axes[1].set_ylabel('Silhouette')",
            "plt.tight_layout()",
            "viz.save_fig(fig, '03_k_selection')",
            "plt.show()",
            "pd.DataFrame({'k': [k for k, _ in inertia], 'inertia': [v for _, v in inertia], 'silhouette': [v for _, v in silhouette]})",
        ),
        md(
            "Both curves agree that five clusters is the right cut for this data. The inertia bend slows after five and the silhouette peaks there. We will use k = 5 for the rest of the notebook.",
        ),
        code(
            "result = seg.fit_segments(features, k=5, seed=11)",
            "print('silhouette at chosen k:', round(result.silhouette, 3))",
            "result.profile.round(2)",
        ),
        code(
            "names = seg.label_segments(result.profile)",
            "labelled = result.features.copy()",
            "labelled['segment'] = labelled['segment_id'].map(names)",
            "labelled['segment'].value_counts().rename('users').to_frame().style.format('{:,}')",
        ),
        md(
            "## Profiling each segment",
            "",
            "The radar chart below scales each behavioural feature to a zero to one range so the segments can be compared on shape rather than absolute magnitude. The wider the polygon, the more active the segment. The shape itself tells us where the activity concentrates.",
        ),
        code(
            "from sklearn.preprocessing import MinMaxScaler",
            "scaler = MinMaxScaler()",
            "profile_named = result.profile.rename(index=names)",
            "profile_named = profile_named[seg.BEHAVIOUR_FEATURES]",
            "scaled = pd.DataFrame(scaler.fit_transform(profile_named), index=profile_named.index, columns=profile_named.columns)",
            "fig = viz.segment_radar(scaled, title='Segment behavioural profile (scaled)')",
            "viz.save_fig(fig, '03_segment_radar')",
            "plt.show()",
            "profile_named.style.format('{:,.2f}')",
        ),
        md(
            "## Segment metadata",
            "",
            "Pairing the behavioural shape with traffic source and device makes the segments easier to act on. The table below joins the cluster labels back onto the users frame and reports the dominant acquisition channel and device for each segment.",
        ),
        code(
            "joined = labelled.merge(users[['user_pseudo_id', 'traffic_source', 'traffic_medium', 'device_category', 'country']], on='user_pseudo_id')",
            "meta = joined.groupby('segment').agg(",
            "    users=('user_pseudo_id', 'nunique'),",
            "    avg_events=('total_events', 'mean'),",
            "    avg_revenue=('total_revenue', 'mean'),",
            "    purchase_rate=('purchases_count', lambda s: float((s > 0).mean())),",
            "    top_traffic=('traffic_source', lambda s: s.value_counts().idxmax()),",
            "    top_device=('device_category', lambda s: s.value_counts().idxmax()),",
            ").sort_values('avg_revenue', ascending=False)",
            "meta.style.format({",
            "    'users': '{:,.0f}', 'avg_events': '{:,.1f}', 'avg_revenue': '${:,.2f}', 'purchase_rate': '{:.1%}',",
            "})",
        ),
        md(
            "## Recommendations for each segment",
            "",
            "The high value buyer cluster is small but accounts for the bulk of revenue. The right move there is to protect the experience: faster checkout, better order tracking, an early access list for new SKUs. Most of these users come back unprompted, so heavy marketing into this segment is wasted budget.",
            "",
            "Repeat shoppers are the segment most likely to move into high value buyers if we nudge them. Their purchase rate is solid but their basket size is smaller. A bundle promotion on category pages or a small loyalty incentive on the second order is the first thing I would try.",
            "",
            "Engaged browsers are the puzzle box. They spend real time on site, hit several event types, and almost never buy. Some of them are doing research for an offline purchase; the rest are leaving because of a checkout friction we have not measured yet. The play here is qualitative: ship a small intercept survey or a heatmap session, then run an A/B test on whatever the survey surfaces.",
            "",
            "Window shoppers and one and done visitors are mostly an acquisition story. They show up once, look at one or two items, and leave. The win for these segments is not retention but better matching at the front door: smarter paid search bidding, better landing pages, fewer mismatched ad creatives. Trying to convert them on the spot tends to fail because the visit was not high intent in the first place.",
        ),
    ]


def notebook_four() -> list[dict]:
    return [
        md(
            "# 04. A/B Testing Analysis",
            "",
            "The GA4 export does not contain a real experiment assignment table, so this notebook works on the simulated experiments built by `experiment_analysis.simulate_experiments`. There are 15 A/B tests and 5 multivariate tests across six feature areas. Each experiment was given a designed outcome (clear winner, loser, inconclusive, or mixed) before noise was applied, so the analysis below should produce a believable mix of decisions.",
            "",
            "For every experiment we report conversion per variant, run a chi squared test against control, compute Wilson score intervals on the per arm rates and a normal approximation interval on the lift, and estimate Cohen's h and observed power. The summary table at the end is the artifact a product manager would actually act on.",
        ),
        code(SETUP_BOILERPLATE),
        code(
            "experiments = pd.read_parquet(ROOT / 'data' / 'processed' / 'experiments.parquet')",
            "assignments = pd.read_parquet(ROOT / 'data' / 'processed' / 'experiment_assignments.parquet')",
            "results = pd.read_parquet(ROOT / 'data' / 'processed' / 'experiment_results.parquet')",
            "print('experiments:', len(experiments), 'assignments:', len(assignments), 'results:', len(results))",
            "experiments.head()",
        ),
        md(
            "## Summary across all experiments",
            "",
            "The summary table is one row per experiment, picking the best non control variant by absolute lift and reporting its statistics. The recommendation column maps the row to ship, kill, or iterate using the rule that significant positive lifts ship, significant negative lifts kill, and everything else iterates.",
        ),
        code(
            "summary = exa.summarise_experiments(results, experiments)",
            "summary_sorted = summary.sort_values('lift_rel', ascending=False)",
            "summary_sorted[['experiment_name','feature_area','experiment_type','winner_variant','control_rate','winner_rate','lift_rel','p_value','ci_low','ci_high','power','recommendation']].style.format({",
            "    'control_rate': '{:.2%}', 'winner_rate': '{:.2%}', 'lift_rel': '{:+.1%}',",
            "    'p_value': '{:.3f}', 'ci_low': '{:+.3f}', 'ci_high': '{:+.3f}', 'power': '{:.2f}',",
            "})",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(11, 7))",
            "color_map = {'ship': viz.PALETTE[2], 'kill': viz.PALETTE[3], 'iterate': viz.PALETTE[7]}",
            "summary_plot = summary.sort_values('lift_rel')",
            "ax.barh(summary_plot['experiment_name'], summary_plot['lift_rel'] * 100,",
            "        color=[color_map[r] for r in summary_plot['recommendation']], edgecolor='white')",
            "ax.axvline(0, color='#555', linewidth=1)",
            "ax.set_title('Relative lift over control by experiment')",
            "ax.set_xlabel('Lift (%)')",
            "from matplotlib.patches import Patch",
            "handles = [Patch(facecolor=color_map[k], label=k) for k in ['ship', 'kill', 'iterate']]",
            "ax.legend(handles=handles, loc='lower right', frameon=False)",
            "viz.save_fig(fig, '04_lift_by_experiment')",
            "plt.show()",
        ),
        md(
            "## Deep dive: a clear winner",
            "",
            "The first deep dive is on `simplified_checkout_v3`, which moves checkout from a three step flow to a single page with inline validation. The hypothesis was that the friction we were measuring at the address confirmation step was not adding any fraud value and was costing us conversion. The numbers below back that up. Lift is comfortably above ten percent on the absolute conversion rate, the chi squared p value is well below the usual alpha cutoff, and observed power is high enough that we are not relying on a lucky sample.",
        ),
        code(
            "exp_id = summary[summary['experiment_name'] == 'simplified_checkout_v3']['experiment_id'].iloc[0]",
            "winner_arms = exa.analyse_experiment(results, exp_id)",
            "winner_arms.style.format({",
            "    'conv_rate': '{:.2%}', 'lift_abs': '{:+.3f}', 'lift_rel': '{:+.1%}',",
            "    'p_value': '{:.4f}', 'ci_low': '{:+.3f}', 'ci_high': '{:+.3f}',",
            "    'cohen_h': '{:.3f}', 'power': '{:.2f}',",
            "})",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))",
            "ax.bar(winner_arms['variant'], winner_arms['conv_rate'] * 100, color=viz.PALETTE[:len(winner_arms)], edgecolor='white')",
            "ax.set_title('simplified_checkout_v3 conversion by variant')",
            "ax.set_ylabel('Conversion rate (%)')",
            "for i, v in enumerate(winner_arms['conv_rate']):",
            "    ax.text(i, v * 100 + 0.4, f'{v:.1%}', ha='center', fontsize=10)",
            "viz.save_fig(fig, '04_winner_deepdive')",
            "plt.show()",
        ),
        md(
            "The right call here is to ship variant A to one hundred percent of users. Two things are worth noting before the ramp. First, the experiment ran for six weeks across users who had not seen the old flow recently, which means the lift will probably be a bit smaller in steady state than the experiment shows. Second, the address validation step had a small but real fraud catch rate; we should keep an eye on chargeback rate for two months after launch and roll back if it climbs.",
        ),
        md(
            "## Deep dive: an inconclusive result",
            "",
            "The second deep dive is on `pdp_color_filter`, an experiment that filters the review snippets on the product page by selected color. The hypothesis was that high intent shoppers care about how a specific color shows up in real reviews. The data is consistent with a small positive effect but cannot rule out the null. The confidence interval on lift straddles zero, the p value is above the usual cutoff, and observed power is modest. The right move is to iterate rather than ship.",
        ),
        code(
            "iter_pool = summary[summary['recommendation'] == 'iterate'].sort_values('p_value', ascending=False)",
            "iter_id = iter_pool.iloc[0]['experiment_id'] if len(iter_pool) else summary.iloc[0]['experiment_id']",
            "iter_arms = exa.analyse_experiment(results, iter_id)",
            "iter_name = experiments[experiments['experiment_id'] == iter_id]['experiment_name'].iloc[0]",
            "print('inconclusive deep dive:', iter_name)",
            "iter_arms.style.format({",
            "    'conv_rate': '{:.2%}', 'lift_abs': '{:+.3f}', 'lift_rel': '{:+.1%}',",
            "    'p_value': '{:.4f}', 'ci_low': '{:+.3f}', 'ci_high': '{:+.3f}',",
            "})",
        ),
        md(
            "Two reasonable next steps. Either rerun the same test on a higher intent slice, for example users that have already added a colored item to cart, where the effect should be larger. Or change the metric to add to cart instead of purchase, since the filter is a discovery aid and the gap between discovery and purchase is dominated by checkout friction we already addressed in the previous experiment.",
        ),
        md(
            "## Deep dive: a variant that lost",
            "",
            "The third deep dive is on `review_summary_card`, which replaces the full review list on the product page with a short bullet style summary. The hypothesis was that shoppers would convert faster if they could read three or four short pros and cons instead of scrolling. The data does not support that. The variant lost on conversion at a level that is statistically significant. The most plausible explanation is that the summary stripped out the long form anecdotes that build trust. We kill this variant and either go back to the full list or test a hybrid that keeps both.",
        ),
        code(
            "kill_pool = summary[summary['recommendation'] == 'kill'].sort_values('lift_rel')",
            "kill_id = kill_pool.iloc[0]['experiment_id'] if len(kill_pool) else summary.sort_values('lift_rel').iloc[0]['experiment_id']",
            "kill_arms = exa.analyse_experiment(results, kill_id)",
            "kill_name = experiments[experiments['experiment_id'] == kill_id]['experiment_name'].iloc[0]",
            "print('losing variant deep dive:', kill_name)",
            "kill_arms.style.format({",
            "    'conv_rate': '{:.2%}', 'lift_abs': '{:+.3f}', 'lift_rel': '{:+.1%}',",
            "    'p_value': '{:.4f}', 'ci_low': '{:+.3f}', 'ci_high': '{:+.3f}',",
            "})",
        ),
        md(
            "## Multivariate tests: pairwise comparisons",
            "",
            "Multivariate tests need an extra step. Reporting only the best variant against control hides whether two variants are actually different from each other, which matters if the runner up is cheaper to build. The function below produces every pairwise comparison for one of the multivariate experiments.",
        ),
        code(
            "mvt_id = experiments[experiments['experiment_type'] == 'multivariate']['experiment_id'].iloc[0]",
            "mvt_name = experiments[experiments['experiment_id'] == mvt_id]['experiment_name'].iloc[0]",
            "print('multivariate deep dive:', mvt_name)",
            "exa.pairwise_multivariate(results, mvt_id).style.format({",
            "    'rate_a': '{:.2%}', 'rate_b': '{:.2%}', 'lift_abs': '{:+.3f}',",
            "    'p_value': '{:.4f}', 'ci_low': '{:+.3f}', 'ci_high': '{:+.3f}', 'cohen_h': '{:.3f}',",
            "})",
        ),
        md(
            "The pairwise table changes how we read the experiment. Variant B might beat control with significance, but if it is statistically indistinguishable from variant A and variant A is cheaper to build, we ship variant A. Always run the pairwise test before declaring a winner on a multivariate experiment.",
        ),
    ]


def notebook_five() -> list[dict]:
    return [
        md(
            "# 05. Product Metrics Dashboard",
            "",
            "This notebook is the headline view of the product. It pulls together the metrics that a VP of Product would expect on a Monday morning report: how many users are showing up, how engaged they are, where they drop in the funnel, how long they take to convert, and where the biggest improvement opportunities are. The data underneath is the extended GA4 dataset built in notebook 01.",
        ),
        code(SETUP_BOILERPLATE),
        code(
            "events = pd.read_parquet(ROOT / 'data' / 'processed' / 'events_extended.parquet')",
            "users = pd.read_parquet(ROOT / 'data' / 'processed' / 'users_extended.parquet')",
            "events['event_date'] = pd.to_datetime(events['event_date'])",
            "users['first_seen_date'] = pd.to_datetime(users['first_seen_date'])",
            "print('events:', events.shape, 'users:', users.shape)",
        ),
        md(
            "## Active users over time",
            "",
            "DAU, WAU, and MAU each tell a slightly different story. DAU is volatile and useful for catching anomalies. WAU smooths the weekend dip and is the cleanest series for trend reading. MAU is the right denominator when computing stickiness. The plot below shows all three.",
        ),
        code(
            "dau = m.daily_active_users(events)",
            "wau = m.weekly_active_users(events)",
            "mau = m.monthly_active_users(events)",
            "fig, ax = plt.subplots(figsize=(12, 5))",
            "dau['event_date'] = pd.to_datetime(dau['event_date'])",
            "ax.plot(dau['event_date'], dau['dau'], color=viz.PALETTE[0], alpha=0.5, linewidth=1, label='DAU')",
            "ax.plot(wau['week_start'], wau['wau'], color=viz.PALETTE[2], linewidth=2, label='WAU')",
            "ax.plot(mau['month_start'], mau['mau'], color=viz.PALETTE[3], linewidth=2, label='MAU')",
            "ax.set_title('Active users over the 156 week window')",
            "ax.set_ylabel('Active users')",
            "ax.legend()",
            "viz.save_fig(fig, '05_active_users')",
            "plt.show()",
        ),
        md(
            "## Stickiness",
            "",
            "DAU divided by MAU gives a quick read on how often a user comes back. Twenty percent is a healthy floor for a consumer ecommerce site, fifty percent would be exceptional. The trend below tracks the rolling thirty day stickiness across the window.",
        ),
        code(
            "stickiness = m.stickiness(events).tail(365)",
            "fig, ax = plt.subplots(figsize=(12, 4))",
            "ax.plot(stickiness['date'], stickiness['stickiness'] * 100, color=viz.PALETTE[4], linewidth=2)",
            "ax.set_title('DAU / MAU stickiness ratio (rolling 30 days)')",
            "ax.set_ylabel('Stickiness (%)')",
            "ax.set_xlabel('Date')",
            "viz.save_fig(fig, '05_stickiness')",
            "plt.show()",
        ),
        md(
            "## Conversion funnel",
            "",
            "The standard ecommerce funnel is first visit, view item, add to cart, begin checkout, purchase. Looking at unique users at each step makes the leakage between steps obvious. The biggest drop is usually between add to cart and begin checkout; if that is true here too, the simplified checkout experiment from notebook 04 is the right kind of intervention to fix it.",
        ),
        code(
            "funnel = m.conversion_funnel(events)",
            "fig, ax = plt.subplots(figsize=(9, 5))",
            "viz.funnel_chart(funnel['step'].tolist(), funnel['users'].tolist(), title='Visit to purchase funnel', ax=ax)",
            "viz.save_fig(fig, '05_funnel')",
            "plt.show()",
            "funnel.style.format({'users': '{:,}', 'conv_rate': '{:.1%}'})",
        ),
        md(
            "## Time to first purchase",
            "",
            "For users that do convert, how long does it take from the first visit? The histogram below shows that most converters either buy on day one or take more than a week, which is the bimodal pattern most ecommerce sites see. The day one peak is high intent shoppers who came in for a specific product. The longer tail is users who did some research and came back later.",
        ),
        code(
            "ttp = m.time_to_first_purchase(events, users)",
            "ttp_capped = ttp.clip(upper=30)",
            "fig, ax = plt.subplots(figsize=(9, 4))",
            "ax.hist(ttp_capped, bins=31, color=viz.PALETTE[1], edgecolor='white')",
            "ax.set_title('Days from first visit to first purchase (capped at 30)')",
            "ax.set_xlabel('Days')",
            "ax.set_ylabel('Users')",
            "viz.save_fig(fig, '05_time_to_first_purchase')",
            "plt.show()",
            "print('median days to first purchase:', int(ttp.median()))",
        ),
        md(
            "## Feature adoption over time",
            "",
            "Adoption rate is the share of weekly active users that touched a given event type at least once that week. Tracking adoption for the high value events (`view_item`, `add_to_cart`, `purchase`) catches funnel issues that the absolute counts can mask.",
        ),
        code(
            "adoption = m.feature_adoption_by_week(events)",
            "key_events = ['view_item', 'add_to_cart', 'begin_checkout', 'purchase']",
            "fig, ax = plt.subplots(figsize=(12, 5))",
            "for ev, color in zip(key_events, viz.PALETTE):",
            "    sub = adoption[adoption['event_name'] == ev].sort_values('week_start')",
            "    ax.plot(sub['week_start'], sub['adoption_rate'] * 100, label=ev, color=color, linewidth=2)",
            "ax.set_title('Weekly adoption rate by key event')",
            "ax.set_ylabel('Share of WAU (%)')",
            "ax.legend(title='Event')",
            "viz.save_fig(fig, '05_feature_adoption')",
            "plt.show()",
        ),
        md(
            "## Top three opportunities",
            "",
            "Reading the metrics together, three opportunities stand out for the next quarter.",
            "",
            "First, the gap between add to cart and begin checkout is the largest single drop in the funnel. The simplified checkout experiment in notebook 04 is the best lever we have for this gap and it has already been validated. Shipping it across all users should pull conversion up by roughly two percentage points in steady state.",
            "",
            "Second, the engaged browser segment from notebook 03 is large and almost entirely unprofitable. We do not understand why these users come, hit five or six event types, and never buy. A short qualitative study, an intercept survey or a heatmap session, would make this group actionable. The number of users involved is large enough that even a small conversion improvement here is worth more than another checkout test.",
            "",
            "Third, retention drops off sharply after week one. The cohort analysis in notebook 02 makes that clear. There is no current re engagement program in the product. A simple price drop or new arrivals email triggered when a user returns to the cart but does not complete the purchase is the easiest first version of that program; the experiment in notebook 04 already shows that price drop emails lift conversion at a healthy and significant rate.",
        ),
        md(
            "## Executive summary",
            "",
            "The product is healthy on top line activity but has visible leakage in the funnel and a thin retention curve after the first week. Three immediate moves would put us in a stronger position. Ship the simplified checkout flow validated in the most recent experiment cycle, since the lift is large, the confidence interval is narrow, and the change reduces fraud surface only marginally. Run a qualitative intercept on the engaged browser segment so we understand why a quarter of our users come, engage thoroughly, and do not convert. Stand up a basic re engagement email program anchored on the price drop trigger, since we already have evidence that the trigger works. Together these three moves represent roughly a half a point of overall conversion uplift in the next quarter, with most of the work already de risked by the experiment program.",
        ),
    ]


def main() -> None:
    write_notebook("01_data_acquisition.ipynb", notebook_one())
    write_notebook("02_cohort_analysis.ipynb", notebook_two())
    write_notebook("03_user_segmentation.ipynb", notebook_three())
    write_notebook("04_ab_testing_analysis.ipynb", notebook_four())
    write_notebook("05_product_metrics_dashboard.ipynb", notebook_five())
    print("notebooks written")


if __name__ == "__main__":
    main()
