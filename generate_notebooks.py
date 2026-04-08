import nbformat as nbf
import os

def create_mmm_notebooks():
    os.makedirs("media_mix_model/notebooks", exist_ok=True)
    
    # 01 Data Exploration
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# MMM Data Exploration\n\nInitial visualization of simulated DTC brand telemetry. We analyze spend correlations, seasonality, and organic trends."))
    nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ndf = pd.read_csv('../data/simulated_data.csv')\ndf.head()"))
    nb.cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(12, 6))\nplt.plot(df['week'], df['conversions'], label='Weekly Conversions')\nplt.title('Weekly Website Conversions (3 Years)')\nplt.legend()\nplt.show()"))
    with open('media_mix_model/notebooks/01_data_exploration.ipynb', 'w') as f:
        nbf.write(nb, f)

    # 02 Model Fitting
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Bayesian MMM Fitting\n\nUsing PyMC-Marketing to estimate media impact with Geometric Adstock and Logistic Saturation."))
    nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\nfrom modeling import MMMWrapper\n\ndf = pd.read_csv('../data/simulated_data.csv')\nmedia_cols = ['linear_tv_spend', 'ctv_spend', 'paid_search_spend', 'paid_social_spend', 'display_spend']\ncontrol_cols = ['sin_period_1', 'cos_period_2', 'competitor_proxy']\n\nwrapper = MMMWrapper(df, 'conversions', 'week', media_cols, control_cols)\nidata = wrapper.fit(tune=1000, draws=1000) # Fast run for demo"))
    with open('media_mix_model/notebooks/02_model_fitting.ipynb', 'w') as f:
        nbf.write(nb, f)

    # 03 Decomposition & Optimization
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Share Decomposition and Budget Optimization\n\nWe decompose estimated conversions into baseline and media-attributed shares, then solve for the optimal budget allocation."))
    with open('media_mix_model/notebooks/03_decomposition_and_optimization.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_tv_notebooks():
    os.makedirs("tv_attribution/notebooks", exist_ok=True)
    
    # 01 Data Exploration
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# TV Attribution: Traffic and Airings Exploration\n\nVisualizing minute-level web telemetry and TV airing logs."))
    nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\n\nsessions = pd.read_csv('../data/sessions.csv', parse_dates=['timestamp'])\nairings = pd.read_csv('../data/airings.csv', parse_dates=['timestamp'])\n\nplt.figure(figsize=(15, 5))\nplt.plot(sessions['timestamp'][:1440], sessions['sessions'][:1440])\nplt.title('Minute-Level Baseline Traffic (Day 1)')\nplt.show()"))
    with open('tv_attribution/notebooks/01_data_exploration.ipynb', 'w') as f:
        nbf.write(nb, f)

    # 02 Spot Attribution
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Spot-Level Lift Estimation\n\nExtracting incremental sessions per TV airing using local linear baselines and parametric response curves."))
    nb.cells.append(nbf.v4.new_code_cell("from attribution import TVAttributor\n\nattr = TVAttributor(sessions, airings)\nresults = attr.run_attribution_pipeline()"))
    with open('tv_attribution/notebooks/02_spot_attribution.ipynb', 'w') as f:
        nbf.write(nb, f)

    # 03 Causal Impact
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Campaign-Level Causal Impact Analysis\n\nUsing BSTS models with control market covariates to estimate aggregate campaign lift."))
    with open('tv_attribution/notebooks/03_causal_impact.ipynb', 'w') as f:
        nbf.write(nb, f)

    # 04 Performance Report
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Performance Monitoring and Scorecards\n\nNetwork-level efficiency analysis, creative effectiveness, and frequency fatigue reports."))
    with open('tv_attribution/notebooks/04_performance_report.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_mmm_notebooks()
    create_tv_notebooks()
    print("Notebook outlines created successfully.")
