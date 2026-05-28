"""Custom CSS styles for the Streamlit dashboard."""

CUSTOM_CSS = """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e94560;
    }
    .stMetric {
        background: rgba(233, 69, 96, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .plot-container {
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: rgba(26, 26, 46, 0.5);
    }
</style>
"""
