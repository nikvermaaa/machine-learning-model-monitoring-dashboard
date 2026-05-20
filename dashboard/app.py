import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brutalist CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Bebas+Neue&display=swap');

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Space Mono', monospace !important;
    background-color: #F2F0E8 !important;
    color: #0A0A0A !important;
}

/* Kill all default Streamlit rounding and softness */
.block-container {
    padding-top: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}

/* Hide keyboard_double_arrow sidebar toggle icon */
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}
button[kind="headerNoPadding"] {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 4px solid #0A0A0A !important;
}
[data-testid="stSidebar"] * {
    color: #F2F0E8 !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stButton button {
    color: #F2F0E8 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #1A1A1A !important;
    border: 2px solid #F2F0E8 !important;
    border-radius: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    background-color: #1A1A1A !important;
    color: #F2F0E8 !important;
}

/* Download + Refresh buttons — black bg, white text */
[data-testid="stSidebar"] .stDownloadButton button,
[data-testid="stSidebar"] .stButton button {
    background-color: #0A0A0A !important;
    color: #F2F0E8 !important;
    border: 2px solid #F2F0E8 !important;
    border-radius: 0 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    padding: 0.5rem !important;
}
[data-testid="stSidebar"] .stDownloadButton button:hover,
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #D4FF00 !important;
    color: #0A0A0A !important;
    border-color: #D4FF00 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #0A0A0A !important;
    border: 3px solid #0A0A0A !important;
    border-radius: 0 !important;
    padding: 1.2rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #888888 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #F2F0E8 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.4rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.7rem !important;
    font-family: 'Space Mono', monospace !important;
}

/* Alert / status boxes */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-width: 3px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 3px solid #0A0A0A !important;
    border-radius: 0 !important;
}

/* Divider */
hr {
    border: 0 !important;
    border-top: 3px solid #0A0A0A !important;
    margin: 1.5rem 0 !important;
}

/* Plotly charts: kill border radius */
.js-plotly-plot {
    border: 3px solid #0A0A0A !important;
}

/* Section headers */
h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.5rem !important;
    letter-spacing: 0.06em !important;
    line-height: 1 !important;
    color: #0A0A0A !important;
    border-bottom: 5px solid #0A0A0A !important;
    padding-bottom: 0.3rem !important;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.6rem !important;
    letter-spacing: 0.08em !important;
    color: #0A0A0A !important;
    border-left: 5px solid #0A0A0A !important;
    padding-left: 0.6rem !important;
    margin-top: 0.5rem !important;
}
h3 {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #555 !important;
    margin-bottom: 0.2rem !important;
}

/* Caption text */
.stCaption, small {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #666 !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH         = "logs/predictions.db"
ALERT_THRESHOLD = 0.4
WARN_THRESHOLD  = 0.2

BRUTALIST_COLORS = {
    "bg":      "#F2F0E8",
    "black":   "#0A0A0A",
    "accent":  "#D4FF00",
    "red":     "#FF2B2B",
    "orange":  "#FF8C00",
    "gray":    "#888888",
}

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(db_path: str):
    conn             = sqlite3.connect(db_path)
    metrics_df       = pd.read_sql("SELECT * FROM weekly_metrics",  conn)
    drift_summary_df = pd.read_sql("SELECT * FROM drift_metrics",   conn)
    feature_drift_df = pd.read_sql("SELECT * FROM feature_drift",   conn)
    conn.close()
    for df in [metrics_df, drift_summary_df, feature_drift_df]:
        df["week"] = df["week"].astype(int)
    return metrics_df, drift_summary_df, feature_drift_df

# ── DB Guard ───────────────────────────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    st.error("DATABASE NOT FOUND at logs/predictions.db")
    st.info("Run simulator then drift detection script, then refresh.")
    st.stop()

try:
    metrics_df, drift_summary_df, feature_drift_df = load_data(DB_PATH)
except Exception as e:
    st.error(f"ERROR LOADING DATABASE: {e}")
    st.stop()

if metrics_df.empty or drift_summary_df.empty or feature_drift_df.empty:
    st.warning("ONE OR MORE TABLES ARE EMPTY. Run simulator and drift scripts first.")
    st.stop()

# ── Safe metric lookup ─────────────────────────────────────────────────────────
def get_metric(df, week, column, label="value"):
    rows = df[df["week"] == week]
    if rows.empty:
        st.error(f"NO DATA FOR {label.upper()} IN WEEK {week}")
        st.stop()
    return rows[column].values[0]

# ── Plotly base layout ─────────────────────────────────────────────────────────
def brutalist_layout(title=""):
    return dict(
        paper_bgcolor="#F2F0E8",
        plot_bgcolor="#F2F0E8",
        font=dict(family="Space Mono, monospace", color="#0A0A0A", size=11),
        title=dict(
            text=title,
            font=dict(family="Space Mono", size=13, color="#0A0A0A"),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#DDDDD0", gridwidth=1,
            linecolor="#0A0A0A", linewidth=2,
            tickfont=dict(family="Space Mono", size=10),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#DDDDD0", gridwidth=1,
            linecolor="#0A0A0A", linewidth=2,
            tickfont=dict(family="Space Mono", size=10),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="#F2F0E8", bordercolor="#0A0A0A", borderwidth=2,
            font=dict(family="Space Mono", size=10),
        ),
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ML MONITOR")
st.sidebar.markdown("---")

available_weeks = sorted(metrics_df["week"].unique())
baseline_week   = available_weeks[0]

selected_week = st.sidebar.selectbox(
    "INSPECT WEEK",
    options=available_weeks,
    index=len(available_weeks) - 1,
)

compare_week = st.sidebar.selectbox(
    "COMPARE AGAINST",
    options=available_weeks,
    index=0,
)

if selected_week == compare_week:
    st.sidebar.warning("SELECT DIFFERENT WEEKS TO COMPARE")

st.sidebar.markdown("---")

report_path = get_metric(drift_summary_df, selected_week, "report_path", "report")
if os.path.exists(report_path):
    with open(report_path, "rb") as f:
        st.sidebar.download_button(
            label=f"DOWNLOAD WEEK {selected_week} REPORT",
            data=f,
            file_name=f"drift_report_week_{selected_week}.html",
            mime="text/html",
        )

st.sidebar.markdown("---")

if st.sidebar.button("FORCE REFRESH"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"BASELINE    : WEEK {baseline_week}")
st.sidebar.caption(f"WEEKS TOTAL : {len(available_weeks)}")
st.sidebar.caption(f"ALERT AT    : {ALERT_THRESHOLD}")
st.sidebar.caption(f"REFRESH     : 60S")

# ── Fetch metrics ──────────────────────────────────────────────────────────────
sel_drift  = get_metric(drift_summary_df, selected_week, "drift_score",       "drift score")
sel_acc    = get_metric(metrics_df,       selected_week, "accuracy",           "accuracy")
sel_f1     = get_metric(metrics_df,       selected_week, "f1_score",           "F1-score")
sel_ndrft  = get_metric(drift_summary_df, selected_week, "n_drifted_features", "drifted features")
sel_ntotal = get_metric(drift_summary_df, selected_week, "n_total_features",   "total features")

cmp_drift  = get_metric(drift_summary_df, compare_week, "drift_score",        "compare drift")
cmp_acc    = get_metric(metrics_df,       compare_week, "accuracy",            "compare accuracy")
cmp_f1     = get_metric(metrics_df,       compare_week, "f1_score",            "compare F1")
cmp_ndrft  = get_metric(drift_summary_df, compare_week, "n_drifted_features",  "compare n_drifted")

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("# ML MODEL MONITOR")
st.caption(f"WEEK {selected_week} vs WEEK {compare_week}  //  {int(sel_ntotal)} FEATURES TRACKED  //  THRESHOLD {ALERT_THRESHOLD}")

# ── Alert Banner ───────────────────────────────────────────────────────────────
if sel_drift >= ALERT_THRESHOLD:
    st.error(
        f"ALERT — WEEK {selected_week} // DRIFT SCORE {sel_drift:.4f} "
        f">= THRESHOLD {ALERT_THRESHOLD} // MODEL RETRAINING RECOMMENDED"
    )
elif sel_drift >= WARN_THRESHOLD:
    st.warning(
        f"WARNING — WEEK {selected_week} // MODERATE DRIFT DETECTED "
        f"// SCORE {sel_drift:.4f} // MONITOR CLOSELY"
    )
else:
    st.success(
        f"HEALTHY — WEEK {selected_week} // SCORE {sel_drift:.4f} "
        f"// NO SIGNIFICANT DRIFT DETECTED"
    )

st.markdown("---")

# ── Metric Cards ───────────────────────────────────────────────────────────────
st.markdown("## PERFORMANCE METRICS")
st.caption(f"WEEK {selected_week} VS WEEK {compare_week}")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "DRIFT SCORE",
    f"{sel_drift:.4f}",
    delta=f"{sel_drift - cmp_drift:+.4f}",
    delta_color="inverse",
)
c2.metric(
    "ACCURACY",
    f"{sel_acc:.4f}",
    delta=f"{sel_acc - cmp_acc:+.4f}",
)
c3.metric(
    "F1 SCORE",
    f"{sel_f1:.4f}",
    delta=f"{sel_f1 - cmp_f1:+.4f}",
)
c4.metric(
    "DRIFTED FEATURES",
    f"{int(sel_ndrft)} / {int(sel_ntotal)}",
    delta=f"{int(sel_ndrft - cmp_ndrft):+d} VS WK {compare_week}",
    delta_color="inverse",
)

st.markdown("---")

# ── Row 1: Charts ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("## DRIFT SCORE OVER TIME")
    st.caption("RISING SCORE INDICATES FEATURE DISTRIBUTION SHIFT")

    sel_point = drift_summary_df[drift_summary_df["week"] == selected_week]
    ds_sorted = drift_summary_df.sort_values("week")

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=ds_sorted["week"], y=ds_sorted["drift_score"],
        mode="lines+markers",
        line=dict(color="#0A0A0A", width=3),
        marker=dict(size=8, color="#0A0A0A", symbol="square"),
        name="DRIFT SCORE",
    ))
    fig_drift.add_trace(go.Scatter(
        x=sel_point["week"], y=sel_point["drift_score"],
        mode="markers",
        marker=dict(
            size=16,
            color=BRUTALIST_COLORS["accent"],
            symbol="square",
            line=dict(color="#0A0A0A", width=2),
        ),
        name=f"WEEK {selected_week}",
    ))
    fig_drift.add_hline(
        y=ALERT_THRESHOLD,
        line_dash="solid",
        line_color=BRUTALIST_COLORS["red"],
        line_width=2,
        annotation_text=f"ALERT {ALERT_THRESHOLD}",
        annotation_font=dict(family="Space Mono", size=10, color=BRUTALIST_COLORS["red"]),
        annotation_position="top left",
    )
    fig_drift.add_hline(
        y=WARN_THRESHOLD,
        line_dash="dash",
        line_color=BRUTALIST_COLORS["orange"],
        line_width=1.5,
        annotation_text=f"WARN {WARN_THRESHOLD}",
        annotation_font=dict(family="Space Mono", size=10, color=BRUTALIST_COLORS["orange"]),
        annotation_position="top left",
    )

    # ── FIX: build layout dict first, then override xaxis to avoid duplicate kwarg ──
    drift_layout = brutalist_layout()
    drift_layout["yaxis_range"] = [0, max(1.0, ds_sorted["drift_score"].max() + 0.1)]
    drift_layout["xaxis"] = dict(
        tickvals=available_weeks,
        tickprefix="WK ",
        showgrid=True,
        gridcolor="#DDDDD0",
        linecolor="#0A0A0A",
        linewidth=2,
        tickfont=dict(family="Space Mono", size=10),
    )
    fig_drift.update_layout(**drift_layout)
    st.plotly_chart(fig_drift, use_container_width=True)

with col_right:
    st.markdown("## ACCURACY & F1 OVER TIME")
    st.caption("FALLING LINES ALONGSIDE RISING DRIFT = MODEL DEGRADATION")

    perf_sorted = metrics_df.sort_values("week")
    sel_perf    = metrics_df[metrics_df["week"] == selected_week]

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=perf_sorted["week"], y=perf_sorted["accuracy"],
        mode="lines+markers",
        line=dict(color="#0A0A0A", width=3),
        marker=dict(size=8, color="#0A0A0A", symbol="square"),
        name="ACCURACY",
    ))
    fig_perf.add_trace(go.Scatter(
        x=perf_sorted["week"], y=perf_sorted["f1_score"],
        mode="lines+markers",
        line=dict(color=BRUTALIST_COLORS["gray"], width=3, dash="dot"),
        marker=dict(size=8, color=BRUTALIST_COLORS["gray"], symbol="diamond"),
        name="F1 SCORE",
    ))
    fig_perf.add_trace(go.Scatter(
        x=sel_perf["week"],
        y=[(sel_perf["accuracy"].values[0] + sel_perf["f1_score"].values[0]) / 2],
        mode="markers",
        marker=dict(
            size=16,
            color=BRUTALIST_COLORS["accent"],
            symbol="square",
            line=dict(color="#0A0A0A", width=2),
        ),
        name=f"WEEK {selected_week}",
    ))

    # ── FIX: same pattern — override xaxis in dict directly ──
    perf_layout = brutalist_layout()
    perf_layout["yaxis_range"] = [0, 1.05]
    perf_layout["xaxis"] = dict(
        tickvals=available_weeks,
        tickprefix="WK ",
        showgrid=True,
        gridcolor="#DDDDD0",
        linecolor="#0A0A0A",
        linewidth=2,
        tickfont=dict(family="Space Mono", size=10),
    )
    fig_perf.update_layout(**perf_layout)
    st.plotly_chart(fig_perf, use_container_width=True)

st.markdown("---")

# ── Row 2: Heatmap + Feature Breakdown ────────────────────────────────────────
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("## FEATURE DRIFT HEATMAP")
    st.caption("P-VALUES PER FEATURE PER WEEK // LOWER = MORE DRIFT // RED = DRIFTED")

    heatmap_df = feature_drift_df.pivot(
        index="feature", columns="week", values="drift_score"
    )
    heatmap_df.columns = [f"WK{w}" for w in heatmap_df.columns]
    heatmap_df = heatmap_df.sort_index()

    styled = heatmap_df.style.background_gradient(
        cmap="RdYlGn",
        axis=None,
        vmin=0.0,
        vmax=0.5,
    ).format("{:.4f}").set_table_styles([
        {"selector": "th", "props": [
            ("font-family", "Space Mono, monospace"),
            ("font-size", "11px"),
            ("font-weight", "700"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.08em"),
            ("background-color", "#0A0A0A"),
            ("color", "#F2F0E8"),
            ("border", "2px solid #0A0A0A"),
        ]},
        {"selector": "td", "props": [
            ("font-family", "Space Mono, monospace"),
            ("font-size", "11px"),
            ("border", "1px solid #DDDDD0"),
        ]},
    ])

    st.dataframe(styled, use_container_width=True, height=420)

with col_right2:
    st.markdown(f"## WEEK {selected_week} // FEATURE BREAKDOWN")
    st.caption("SORTED BY DRIFT SEVERITY // LOWER P-VALUE = MORE DRIFTED")

    week_features = feature_drift_df[
        feature_drift_df["week"] == selected_week
    ].copy().sort_values("drift_score")

    week_features["STATUS"] = week_features["is_drifted"].apply(
        lambda x: "DRIFTED" if x == 1 else "OK"
    )

    display_df = week_features[["feature", "drift_score", "stat_test", "STATUS"]].rename(columns={
        "feature":     "FEATURE",
        "drift_score": "P-VALUE",
        "stat_test":   "TEST",
    }).reset_index(drop=True)

    def highlight_drifted(row):
        if row["STATUS"] == "DRIFTED":
            return ["background-color: #FF2B2B; color: #F2F0E8; font-weight: 700"] * len(row)
        return [""] * len(row)

    styled_features = display_df.style.apply(highlight_drifted, axis=1).format(
        {"P-VALUE": "{:.6f}"}
    ).set_table_styles([
        {"selector": "th", "props": [
            ("font-family", "Space Mono, monospace"),
            ("font-size", "11px"),
            ("font-weight", "700"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.08em"),
            ("background-color", "#0A0A0A"),
            ("color", "#F2F0E8"),
            ("border", "2px solid #0A0A0A"),
        ]},
        {"selector": "td", "props": [
            ("font-family", "Space Mono, monospace"),
            ("font-size", "11px"),
            ("border", "1px solid #DDDDD0"),
        ]},
    ])

    st.dataframe(styled_features, use_container_width=True, height=420)

st.markdown("---")

# ── Row 3: Drifted Features Bar Chart ─────────────────────────────────────────
st.markdown("## DRIFTED FEATURES COUNT PER WEEK")
st.caption(f"THRESHOLD FIRES AT {int(sel_ntotal * ALERT_THRESHOLD)} / {int(sel_ntotal)} FEATURES")

ds_bar = drift_summary_df.sort_values("week")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=[f"WK{w}" for w in ds_bar["week"]],
    y=ds_bar["n_drifted_features"],
    marker=dict(
        color=[
            BRUTALIST_COLORS["red"] if row["is_drifted"] == 1
            else BRUTALIST_COLORS["black"]
            for _, row in ds_bar.iterrows()
        ],
        line=dict(color="#0A0A0A", width=2),
    ),
    text=ds_bar["n_drifted_features"].astype(int),
    textposition="outside",
    textfont=dict(family="Space Mono", size=13, color="#0A0A0A"),
    name="DRIFTED FEATURES",
))
fig_bar.add_hline(
    y=sel_ntotal * ALERT_THRESHOLD,
    line_dash="solid",
    line_color=BRUTALIST_COLORS["red"],
    line_width=2,
    annotation_text="ALERT THRESHOLD",
    annotation_font=dict(family="Space Mono", size=10, color=BRUTALIST_COLORS["red"]),
    annotation_position="top left",
)
fig_bar.update_layout(
    **brutalist_layout(),
    yaxis_range=[0, int(sel_ntotal) + 2],
    bargap=0.25,
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"EVIDENTLY AI + STREAMLIT + SQLITE  //  "
    f"ALERT THRESHOLD {ALERT_THRESHOLD}  //  "
    f"DATA REFRESHES EVERY 60S"
)