import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import os
import sys
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Audio Landscape",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@400;700&display=swap');

    :root {
        --bg:        #000000;
        --surface:   #111111;
        --surface2:  #1a1a1a;
        --surface3:  #222222;
        --green:     #1DB954;
        --green-dim: #158a3e;
        --green-glow:rgba(29,185,84,0.15);
        --text:      #FFFFFF;
        --muted:     #a7a7a7;
        --accent:    #282828;
        --border:    #2a2a2a;
    }

    /* ── Base ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] { background: var(--bg) !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

    /* ── Sidebar label overrides ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption { color: var(--muted) !important; font-size: 11px !important; }

    /* ── Slider track & thumb ── */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: var(--green) !important;
        border-color: var(--green) !important;
    }
    [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    [data-testid="stSlider"] div[data-testid="stTickBarMax"] { color: var(--muted) !important; }

    /* ── Multiselect tags ── */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: var(--green) !important;
        color: #000 !important;
        font-weight: 600;
        border-radius: 20px;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] span { color: #000 !important; }

    /* ── Selectbox / multiselect dropdown ── */
    [data-testid="stMultiSelect"] > div > div,
    [data-testid="stSelectbox"] > div > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color .2s;
    }
    div[data-testid="metric-container"]:hover { border-color: var(--green); }
    div[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--green) !important; font-size: 26px; font-weight: 700; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--surface2);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 13px;
        transition: color .15s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); }
    .stTabs [aria-selected="true"] {
        background: var(--green) !important;
        color: #000 !important;
        font-weight: 700 !important;
    }
    /* First character of each tab = the symbol — color it green on inactive tabs */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) p::first-letter {
        color: var(--green);
        font-size: 15px;
    }

    /* ── Section typography ── */
    .section-header { font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 2px; }
    .section-sub    { font-size: 13px; color: var(--muted); margin-bottom: 16px; }
    h1 { color: #fff !important; }

    /* ── Sidebar brand block ── */
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 28px 20px 20px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 8px;
    }
    .sb-brand-icon {
        width: 36px; height: 36px;
        background: var(--green);
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
    }
    .sb-brand-text { line-height: 1.2; }
    .sb-brand-title { font-size: 15px; font-weight: 700; color: #fff; }
    .sb-brand-sub   { font-size: 11px; color: var(--muted); }

    /* ── Sidebar section label ── */
    .sb-section {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
        color: var(--muted);
        padding: 16px 20px 6px;
    }

    /* ── Sidebar stats pill row ── */
    .sb-stats {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 16px 20px 24px;
        border-top: 1px solid var(--border);
        margin-top: 12px;
    }
    .sb-stat {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .sb-stat:hover { border-color: var(--green); }
    .sb-stat-val { font-size: 26px; font-weight: 800; color: var(--green); letter-spacing: -0.5px; }
    .sb-stat-lbl { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }

    /* ── Sidebar padding shim ── */
    [data-testid="stSidebar"] section.main > div { padding: 0 !important; }
    [data-testid="stSidebar"] .block-container { padding: 0 !important; }

    /* streamlit inner padding for controls */
    [data-testid="stSidebar"] .element-container { padding-left: 16px; padding-right: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
CSV_FILENAME = "spotify_updated.csv"

def find_csv():
    """Search for the CSV next to the script, or in cwd."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, CSV_FILENAME),
        os.path.join(os.getcwd(), CSV_FILENAME),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["mode_label"] = df["mode"].map({1: "Major", 0: "Minor"})
    df["explicit_label"] = df["explicit"].map({True: "Explicit", False: "Clean"})
    df["popularity_tier"] = pd.cut(
        df["popularity"],
        bins=[-1, 20, 40, 60, 80, 100],
        labels=["Very Low (0–20)", "Low (21–40)", "Medium (41–60)", "High (61–80)", "Very High (81–100)"],
    )
    df["duration_min"] = df["duration_ms"] / 60000
    return df


# ── Locate CSV (with friendly fallback uploader) ─────────────────────────────
csv_path = find_csv()

if csv_path is None:
    st.markdown("<h1 style='font-size:32px; font-weight:800; color:#fff;'><span style='color:#1DB954;'>♪</span> Spotify Audio Landscape</h1>", unsafe_allow_html=True)
    st.warning("**CSV not found.** Place `spotify_updated.csv` in the same folder as this script, or upload it below.")
    uploaded = st.file_uploader("Upload spotify_updated.csv", type="csv")
    if uploaded is not None:
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        shutil.copyfileobj(uploaded, tmp)
        tmp.close()
        csv_path = tmp.name
    else:
        st.stop()

df = load_data(csv_path)

AUDIO_FEATURES = ["danceability", "energy", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence"]
SPOTIFY_GREEN  = "#1DB954"
PLOTLY_TEMPLATE = "plotly_dark"

GENRE_COLOR_SEQ = px.colors.qualitative.Vivid + px.colors.qualitative.Pastel

def card(content_fn, *args, **kwargs):
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        content_fn(*args, **kwargs)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon" style="background:#1DB954;color:#000;font-size:20px;font-weight:900;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;">♪</div>
        <div class="sb-brand-text">
            <div class="sb-brand-title">Audio Landscape</div>
            <div class="sb-brand-sub">113,842 tracks · 114 genres</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Genres
    st.markdown('<div class="sb-section">Genres</div>', unsafe_allow_html=True)
    all_genres = sorted(df["track_genre"].unique())
    default_genres = ["pop", "rock", "hip-hop", "jazz", "classical", "electronic",
                      "k-pop", "metal", "r-n-b", "acoustic"]
    default_genres = [g for g in default_genres if g in all_genres]
    selected_genres = st.multiselect(
        "Filter Genres",
        options=all_genres,
        default=default_genres,
        label_visibility="collapsed",
    )

    # Popularity
    st.markdown('<div class="sb-section">Popularity</div>', unsafe_allow_html=True)
    pop_range = st.slider("Popularity Range", 0, 100, (0, 100), label_visibility="collapsed")

    # Bottom stats
    st.markdown("""
    <div class="sb-stats">
        <div class="sb-stat">
            <div class="sb-stat-val">114k</div>
            <div class="sb-stat-lbl">Tracks</div>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-val">114</div>
            <div class="sb-stat-lbl">Genres</div>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-val">7</div>
            <div class="sb-stat-lbl">Features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
if selected_genres:
    fdf = df[df["track_genre"].isin(selected_genres)]
else:
    fdf = df.copy()
fdf = fdf[(fdf["popularity"] >= pop_range[0]) & (fdf["popularity"] <= pop_range[1])]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:36px; font-weight:800; margin-bottom:0;">
    <span style="color:#1DB954;">♪</span> Spotify Audio Landscape
</h1>
<p style="color:#b3b3b3; font-size:15px; margin-top:4px; margin-bottom:24px;">
    Exploring what makes a track succeed across 114,000 songs · {n} genres selected · popularity {lo}–{hi}
</p>
""".format(n=len(selected_genres) if selected_genres else 114,
           lo=pop_range[0], hi=pop_range[1]),
unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tracks", f"{len(fdf):,}")
k2.metric("Avg Popularity", f"{fdf['popularity'].mean():.1f}")
k3.metric("Avg Energy",     f"{fdf['energy'].mean():.2f}")
k4.metric("Avg Danceability", f"{fdf['danceability'].mean():.2f}")
k5.metric("Explicit %",    f"{fdf['explicit'].mean()*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "◈  Genre Overview",
    "◎  Feature Deep-Dive",
    "◆  Popularity Drivers",
    "◉  Mood & Mode",
    "◐  Explicit Content",
    "⬡  Audio Clusters",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Genre Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Genre Popularity Rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Which genres produce the most popular tracks on average?</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        genre_pop = (fdf.groupby("track_genre")["popularity"]
                     .agg(["mean", "median", "std", "count"])
                     .reset_index()
                     .rename(columns={"mean": "Avg Popularity", "median": "Median",
                                      "std": "Std Dev", "count": "Track Count"}))
        genre_pop = genre_pop.sort_values("Avg Popularity", ascending=True)

        fig = px.bar(
            genre_pop, x="Avg Popularity", y="track_genre",
            orientation="h",
            color="Avg Popularity",
            color_continuous_scale=["#1a1a2e", "#1DB954"],
            hover_data={"Median": True, "Std Dev": ":.2f", "Track Count": True},
            template=PLOTLY_TEMPLATE,
            height=max(400, len(genre_pop) * 26),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=10, b=10),
            yaxis_title="", xaxis_title="Average Popularity Score",
            font=dict(color="#b3b3b3"),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Radar: Avg Audio Features by Genre**")
        top_genres_radar = (fdf.groupby("track_genre")["popularity"]
                            .mean().sort_values(ascending=False).head(8).index.tolist())
        radar_df = fdf[fdf["track_genre"].isin(top_genres_radar)].groupby("track_genre")[AUDIO_FEATURES].mean()

        fig2 = go.Figure()
        for i, genre in enumerate(radar_df.index):
            vals = radar_df.loc[genre].tolist()
            vals += [vals[0]]
            fig2.add_trace(go.Scatterpolar(
                r=vals,
                theta=AUDIO_FEATURES + [AUDIO_FEATURES[0]],
                fill="toself",
                opacity=0.35,
                name=genre,
                line=dict(width=2),
            ))
        fig2.update_layout(
            polar=dict(
                bgcolor="rgba(30,30,30,0.8)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333", tickcolor="#333"),
                angularaxis=dict(gridcolor="#333", tickcolor="#b3b3b3"),
            ),
            showlegend=True,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b3b3b3", size=11),
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(bgcolor="rgba(24,24,24,0.9)", font=dict(size=13, color="#ffffff"), bordercolor="#333", borderwidth=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Audio Feature Heatmap Across Genres</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average values for each audio feature — spot genre personality at a glance</div>', unsafe_allow_html=True)

    heat_df = fdf.groupby("track_genre")[AUDIO_FEATURES].mean()
    # Normalize each column 0-1 for visual comparison
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min())

    fig3 = px.imshow(
        heat_norm.T,
        color_continuous_scale=["#0d0d0d", "#1DB954"],
        aspect="auto",
        template=PLOTLY_TEMPLATE,
        zmin=0, zmax=1,
        labels=dict(color="Relative Score"),
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b3b3b3"),
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="Normalized", tickfont=dict(color="#b3b3b3")),
        height=320,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Feature Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Audio Feature Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How do sonic characteristics spread across all tracks in the selected genres?</div>', unsafe_allow_html=True)

    # Distribution grid
    fig_dist = make_subplots(rows=2, cols=4,
                              subplot_titles=AUDIO_FEATURES + ["tempo (normalized)"],
                              vertical_spacing=0.15, horizontal_spacing=0.08)
    all_features_plot = AUDIO_FEATURES + ["tempo"]
    for i, feat in enumerate(all_features_plot):
        row, col = divmod(i, 4)
        vals = fdf[feat].dropna()
        if feat == "tempo":
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        fig_dist.add_trace(
            go.Histogram(x=vals, nbinsx=50, marker_color=SPOTIFY_GREEN,
                         marker_line_width=0, opacity=0.85, name=feat, showlegend=False),
            row=row + 1, col=col + 1,
        )
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template=PLOTLY_TEMPLATE, height=480,
        font=dict(color="#b3b3b3", size=10),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    fig_dist.update_xaxes(gridcolor="#2a2a2a", zerolinecolor="#333")
    fig_dist.update_yaxes(gridcolor="#2a2a2a", zerolinecolor="#333", showticklabels=True)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Feature vs. Feature Scatter</div>', unsafe_allow_html=True)

    scat_col1, scat_col2, scat_spacer = st.columns([1, 1, 3])
    with scat_col1:
        x_feature = st.selectbox("X-axis feature", AUDIO_FEATURES, index=1, key="scatter_x")
    with scat_col2:
        y_feature = st.selectbox("Y-axis feature", AUDIO_FEATURES, index=0, key="scatter_y")

    st.markdown(f'<div class="section-sub">Exploring the relationship between <b>{x_feature}</b> and <b>{y_feature}</b> — colored by genre</div>', unsafe_allow_html=True)

    sample = fdf.sample(min(6000, len(fdf)), random_state=42)
    fig_scat = px.scatter(
        sample, x=x_feature, y=y_feature,
        color="track_genre",
        color_discrete_sequence=GENRE_COLOR_SEQ,
        opacity=0.55, size_max=5,
        hover_data={"track_name": True, "artists": True, "popularity": True,
                    "track_genre": True, x_feature: True, y_feature: True},
        template=PLOTLY_TEMPLATE,
        height=520,
        labels={x_feature: x_feature.capitalize(), y_feature: y_feature.capitalize()},
    )
    fig_scat.update_traces(marker=dict(size=4))
    fig_scat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b3b3b3"),
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(bgcolor="rgba(18,18,18,0.95)", font=dict(size=12, color="#ffffff"),
                    title=dict(text="Genre", font=dict(size=13, color="#1DB954")),
                    itemsizing="constant", bordercolor="#333", borderwidth=1),
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Genre Feature Box Plots</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Spread and median of a chosen feature across genres</div>', unsafe_allow_html=True)
    box_feat = st.selectbox("Select feature for box plot", AUDIO_FEATURES, index=1)
    genre_order = (fdf.groupby("track_genre")[box_feat].median()
                   .sort_values(ascending=False).index.tolist())
    fig_box = px.box(
        fdf, x="track_genre", y=box_feat,
        color="track_genre",
        color_discrete_sequence=GENRE_COLOR_SEQ,
        category_orders={"track_genre": genre_order},
        template=PLOTLY_TEMPLATE, height=460,
        labels={"track_genre": "", box_feat: box_feat.capitalize()},
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(color="#b3b3b3"),
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(tickangle=-45, gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Popularity Drivers
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">What Drives Popularity?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Correlation between audio features and popularity score</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        corr_features = AUDIO_FEATURES + ["tempo", "loudness", "duration_min"]
        corr_vals = fdf[corr_features + ["popularity"]].corr()["popularity"].drop("popularity").sort_values()
        colors = [SPOTIFY_GREEN if v > 0 else "#e74c3c" for v in corr_vals.values]

        fig_corr = go.Figure(go.Bar(
            x=corr_vals.values,
            y=corr_vals.index,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
        ))
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            template=PLOTLY_TEMPLATE, height=420,
            font=dict(color="#b3b3b3"),
            title=dict(text="Pearson Correlation with Popularity", font=dict(size=14, color="#fff")),
            margin=dict(l=0, r=0, t=40, b=10),
            xaxis=dict(gridcolor="#2a2a2a", zeroline=True, zerolinecolor="#555", title="Correlation"),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        pop_feat = st.selectbox("Feature vs. Popularity", AUDIO_FEATURES + ["tempo", "loudness"], index=0)
        sample2 = fdf.sample(min(5000, len(fdf)), random_state=7)
        fig_pop = px.scatter(
            sample2, x=pop_feat, y="popularity",
            color="track_genre",
            color_discrete_sequence=GENRE_COLOR_SEQ,
            opacity=0.5,
            trendline=None,
            hover_data={"track_name": True, "artists": True, "popularity": True},
            template=PLOTLY_TEMPLATE, height=420,
            labels={pop_feat: pop_feat.capitalize(), "popularity": "Popularity Score"},
        )
        fig_pop.update_traces(marker=dict(size=4))
        fig_pop.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b3b3b3"),
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(bgcolor="rgba(18,18,18,0.95)", font=dict(size=12, color="#ffffff"),
                    title=dict(text="Genre", font=dict(size=13, color="#1DB954")),
                    itemsizing="constant", bordercolor="#333", borderwidth=1),
            xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_pop, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Audio Profile by Popularity Tier</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How does each audio feature trend as popularity rises?</div>', unsafe_allow_html=True)

    tier_selected_feats = st.multiselect(
        "Select features to compare",
        options=AUDIO_FEATURES,
        default=["danceability", "energy", "acousticness", "valence"],
        key="tier_features",
    )

    TIER_ORDER = ["Very Low (0–20)", "Low (21–40)", "Medium (41–60)", "High (61–80)", "Very High (81–100)"]
    FEATURE_COLORS = {
        "danceability":      "#1DB954",
        "energy":            "#f59e0b",
        "speechiness":       "#e74c3c",
        "acousticness":      "#4a9eff",
        "instrumentalness":  "#a78bfa",
        "liveness":          "#f97316",
        "valence":           "#ec4899",
    }

    tier_df = (fdf.groupby("popularity_tier", observed=True)[AUDIO_FEATURES]
               .mean()
               .reindex([t for t in TIER_ORDER if t in fdf["popularity_tier"].cat.categories]))

    fig_tier = go.Figure()
    for feat in (tier_selected_feats or AUDIO_FEATURES):
        if feat not in tier_df.columns:
            continue
        fig_tier.add_trace(go.Scatter(
            x=tier_df.index.tolist(),
            y=tier_df[feat].values,
            mode="lines+markers",
            name=feat.capitalize(),
            line=dict(color=FEATURE_COLORS.get(feat, "#888"), width=2.5),
            marker=dict(size=8, color=FEATURE_COLORS.get(feat, "#888"),
                        line=dict(color="#0d0d0d", width=1.5)),
            hovertemplate="<b>" + feat.capitalize() + "</b><br>%{x}: %{y:.3f}<extra></extra>",
        ))

    fig_tier.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template=PLOTLY_TEMPLATE, height=420,
        font=dict(color="#b3b3b3"),
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(gridcolor="#2a2a2a", title="Popularity Tier"),
        yaxis=dict(gridcolor="#2a2a2a", title="Average Score", range=[0, 1]),
        legend=dict(bgcolor="rgba(18,18,18,0.95)", font=dict(size=13, color="#ffffff"),
                    orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bordercolor="#333", borderwidth=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Top 50 Tracks by Popularity</div>', unsafe_allow_html=True)
    top50 = (fdf.nlargest(50, "popularity")
             [["track_name", "artists", "track_genre", "popularity",
               "energy", "danceability", "valence", "acousticness"]]
             .reset_index(drop=True))
    top50.index += 1
    def color_popularity(val):
        """Green shade scaled to popularity value — no matplotlib needed."""
        intensity = int(val / 100 * 180)
        return f"background-color: rgba(29, {80 + intensity}, 84, 0.55); color: #fff"

    st.dataframe(
        top50.style.map(color_popularity, subset=["popularity"])
                   .format({"energy": "{:.2f}", "danceability": "{:.2f}",
                            "valence": "{:.2f}", "acousticness": "{:.2f}"}),
        use_container_width=True, height=420,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Mood & Mode
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Major vs. Minor: The Mood of Music</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Mode distribution across genres and its relationship with valence (emotional positivity)</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        mode_genre = (fdf.groupby(["track_genre", "mode_label"])
                      .size().reset_index(name="count"))
        mode_total = mode_genre.groupby("track_genre")["count"].transform("sum")
        mode_genre["pct"] = mode_genre["count"] / mode_total * 100
        mode_major = mode_genre[mode_genre["mode_label"] == "Major"].sort_values("pct", ascending=True)

        fig_mode = px.bar(
            mode_major, x="pct", y="track_genre",
            orientation="h",
            color="pct",
            color_continuous_scale=["#c0392b", "#e67e22", "#1DB954"],
            template=PLOTLY_TEMPLATE, height=max(380, len(mode_major) * 28),
            labels={"pct": "% Major Key", "track_genre": ""},
        )
        fig_mode.add_vline(x=50, line_dash="dot", line_color="#888",
                           annotation_text="50%", annotation_font_color="#888")
        fig_mode.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(color="#b3b3b3"),
            title=dict(text="% of Tracks in Major Key", font=dict(size=14, color="#fff")),
            margin=dict(l=0, r=0, t=40, b=10),
            xaxis=dict(gridcolor="#2a2a2a", range=[0, 100]),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with col_m2:
        # Valence by mode and popularity tier
        val_mode = fdf.groupby(["mode_label", "popularity_tier"], observed=True)["valence"].mean().reset_index()
        fig_val = px.bar(
            val_mode, x="popularity_tier", y="valence", color="mode_label",
            barmode="group",
            color_discrete_map={"Major": SPOTIFY_GREEN, "Minor": "#e74c3c"},
            template=PLOTLY_TEMPLATE, height=380,
            labels={"valence": "Avg Valence", "popularity_tier": "Popularity Tier", "mode_label": "Mode"},
        )
        fig_val.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b3b3b3"),
            title=dict(text="Valence by Mode & Popularity Tier", font=dict(size=14, color="#fff")),
            margin=dict(l=0, r=0, t=40, b=10),
            xaxis=dict(gridcolor="#2a2a2a", tickangle=-20),
            yaxis=dict(gridcolor="#2a2a2a"),
            legend=dict(bgcolor="rgba(18,18,18,0.95)", font=dict(size=13, color="#ffffff"),
                        title=dict(text="Mode", font=dict(size=13, color="#1DB954")),
                        bordercolor="#333", borderwidth=1),
        )
        st.plotly_chart(fig_val, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Mood Quadrant Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Share of tracks per emotional bucket across genres — split by valence (happy/sad) and energy (calm/intense)</div>', unsafe_allow_html=True)

    # Assign each track to a mood quadrant
    mood_tmp = fdf[["track_genre", "valence", "energy"]].dropna().copy()
    mood_tmp["quadrant"] = np.select(
        [
            (mood_tmp["valence"] <  0.5) & (mood_tmp["energy"] <  0.5),
            (mood_tmp["valence"] <  0.5) & (mood_tmp["energy"] >= 0.5),
            (mood_tmp["valence"] >= 0.5) & (mood_tmp["energy"] <  0.5),
            (mood_tmp["valence"] >= 0.5) & (mood_tmp["energy"] >= 0.5),
        ],
        ["😢 Sad / Calm", "⚡ Angry / Intense", "🌿 Peaceful / Chill", "🎉 Happy / Energetic"],
        default="Unknown",
    )

    QUADRANT_COLORS = {
        "😢 Sad / Calm":        "#5b8dd9",
        "⚡ Angry / Intense":   "#e74c3c",
        "🌿 Peaceful / Chill":  "#1DB954",
        "🎉 Happy / Energetic": "#f39c12",
    }
    QUADRANT_ORDER = ["🎉 Happy / Energetic", "🌿 Peaceful / Chill", "⚡ Angry / Intense", "😢 Sad / Calm"]

    # Overall donut + per-genre stacked bar side by side
    col_mq1, col_mq2 = st.columns([1, 3])

    with col_mq1:
        overall_q = mood_tmp["quadrant"].value_counts().reset_index()
        overall_q.columns = ["quadrant", "count"]
        overall_q["pct"] = overall_q["count"] / overall_q["count"].sum() * 100
        fig_donut = go.Figure(go.Pie(
            labels=overall_q["quadrant"],
            values=overall_q["count"],
            marker_colors=[QUADRANT_COLORS[q] for q in overall_q["quadrant"]],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(size=11, color="#fff"),
            sort=False,
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
            height=320,
            title=dict(text="Overall Split", font=dict(size=13, color="#fff"), x=0.5),
            font=dict(color="#b3b3b3"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_mq2:
        # Per-genre counts → percentage stacked bar
        genre_q = (mood_tmp.groupby(["track_genre", "quadrant"])
                   .size().reset_index(name="count"))
        genre_totals = genre_q.groupby("track_genre")["count"].transform("sum")
        genre_q["pct"] = genre_q["count"] / genre_totals * 100

        # Sort genres by % Happy / Energetic descending
        happy_pct = (genre_q[genre_q["quadrant"] == "🎉 Happy / Energetic"]
                     .set_index("track_genre")["pct"])
        genre_order_mood = happy_pct.sort_values(ascending=True).index.tolist()

        fig_mood = go.Figure()
        for quad in QUADRANT_ORDER:
            sub = genre_q[genre_q["quadrant"] == quad].set_index("track_genre").reindex(genre_order_mood).fillna(0)
            fig_mood.add_trace(go.Bar(
                name=quad,
                x=sub["pct"].values,
                y=genre_order_mood,
                orientation="h",
                marker_color=QUADRANT_COLORS[quad],
                marker_line_width=0,
                hovertemplate=f"<b>%{{y}}</b><br>{quad}: %{{x:.1f}}%<extra></extra>",
            ))
        fig_mood.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            template=PLOTLY_TEMPLATE,
            height=max(400, len(genre_order_mood) * 26),
            font=dict(color="#b3b3b3"),
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(title="% of Tracks", gridcolor="#2a2a2a", range=[0, 100],
                       ticksuffix="%"),
            yaxis=dict(gridcolor="#2a2a2a"),
            legend=dict(bgcolor="rgba(18,18,18,0.95)", font=dict(size=13, color="#ffffff"),
                        orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        bordercolor="#333", borderwidth=1),
        )
        st.plotly_chart(fig_mood, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explicit Content
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Explicit Content Across Genres</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How explicit tracks distribute across genres, and whether they trend differently in popularity</div>', unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([2, 1])

    with col_e1:
        exp_genre = (fdf.groupby(["track_genre", "explicit_label"])
                     .size().reset_index(name="count"))
        exp_total = exp_genre.groupby("track_genre")["count"].transform("sum")
        exp_genre["pct"] = exp_genre["count"] / exp_total * 100
        exp_pct = (exp_genre[exp_genre["explicit_label"] == "Explicit"]
                   .sort_values("pct", ascending=True))

        fig_exp = px.bar(
            exp_pct, x="pct", y="track_genre",
            orientation="h",
            color="pct",
            color_continuous_scale=["#1a1a2e", "#e74c3c"],
            template=PLOTLY_TEMPLATE,
            height=max(380, len(exp_pct) * 28),
            labels={"pct": "% Explicit Tracks", "track_genre": ""},
        )
        fig_exp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, font=dict(color="#b3b3b3"),
            title=dict(text="% Explicit Tracks by Genre", font=dict(size=14, color="#fff")),
            margin=dict(l=0, r=0, t=40, b=10),
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_exp, use_container_width=True)

    with col_e2:
        total_exp = fdf["explicit"].sum()
        total_clean = len(fdf) - total_exp
        fig_pie = go.Figure(go.Pie(
            labels=["Clean", "Explicit"],
            values=[total_clean, total_exp],
            marker_colors=["#1DB954", "#e74c3c"],
            hole=0.55,
            textinfo="label+percent",
            textfont=dict(color="#fff", size=13),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            template=PLOTLY_TEMPLATE,
            showlegend=False,
            margin=dict(l=10, r=10, t=60, b=10),
            height=300,
            title=dict(text="Overall Split", font=dict(size=14, color="#fff"), x=0.5),
            font=dict(color="#b3b3b3"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Avg popularity by explicit
        exp_pop = fdf.groupby("explicit_label")["popularity"].mean().reset_index()
        for _, row in exp_pop.iterrows():
            label = "● Explicit" if row["explicit_label"] == "Explicit" else "● Clean"
            st.metric(label, f"{row['popularity']:.1f} avg pop")

    st.markdown("---")
    st.markdown('<div class="section-header">Explicit vs. Clean: Audio Feature Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Do explicit tracks have a distinct audio fingerprint?</div>', unsafe_allow_html=True)

    exp_feat = fdf.groupby("explicit_label")[AUDIO_FEATURES].mean()
    # Build dumbbell: one row per feature, two endpoints + connecting line
    feat_order = sorted(
        AUDIO_FEATURES,
        key=lambda f: abs(
            exp_feat.loc["Explicit", f] - exp_feat.loc["Clean", f]
            if "Explicit" in exp_feat.index and "Clean" in exp_feat.index else 0
        ),
        reverse=True,
    )
    fig_expfeat = go.Figure()
    for feat in feat_order:
        v_clean    = exp_feat.loc["Clean",    feat] if "Clean"    in exp_feat.index else 0
        v_explicit = exp_feat.loc["Explicit", feat] if "Explicit" in exp_feat.index else 0
        # Connector line
        fig_expfeat.add_trace(go.Scatter(
            x=[v_clean, v_explicit], y=[feat, feat],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=3),
            showlegend=False, hoverinfo="skip",
        ))
        # Clean dot
        fig_expfeat.add_trace(go.Scatter(
            x=[v_clean], y=[feat],
            mode="markers",
            marker=dict(color=SPOTIFY_GREEN, size=14, line=dict(color="#fff", width=1.5)),
            name="Clean", legendgroup="Clean",
            showlegend=(feat == feat_order[0]),
            hovertemplate=f"<b>{feat}</b><br>Clean: %{{x:.3f}}<extra></extra>",
        ))
        # Explicit dot
        fig_expfeat.add_trace(go.Scatter(
            x=[v_explicit], y=[feat],
            mode="markers",
            marker=dict(color="#e74c3c", size=14, line=dict(color="#fff", width=1.5)),
            name="Explicit", legendgroup="Explicit",
            showlegend=(feat == feat_order[0]),
            hovertemplate=f"<b>{feat}</b><br>Explicit: %{{x:.3f}}<extra></extra>",
        ))
    fig_expfeat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        template=PLOTLY_TEMPLATE, height=400,
        font=dict(color="#b3b3b3"),
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(title="Average Score", gridcolor="#2a2a2a", range=[0, 1]),
        yaxis=dict(gridcolor="#2a2a2a", categoryorder="array", categoryarray=feat_order[::-1]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_expfeat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Audio Clusters (PCA + KMeans)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Audio Clustering via K-Means</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Grouping tracks by their sonic fingerprint — regardless of genre label</div>', unsafe_allow_html=True)

    @st.cache_data
    def compute_clusters(genres_key, pop_lo, pop_hi, n_cl):
        sub = df.copy()
        if genres_key:
            sub = sub[sub["track_genre"].isin(genres_key)]
        sub = sub[(sub["popularity"] >= pop_lo) & (sub["popularity"] <= pop_hi)]
        sub = sub.dropna(subset=AUDIO_FEATURES).sample(min(10000, len(sub)), random_state=42)

        scaler = StandardScaler()
        X = scaler.fit_transform(sub[AUDIO_FEATURES])

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)

        km = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        sub = sub.copy()
        sub["PC1"] = coords[:, 0]
        sub["PC2"] = coords[:, 1]
        sub["Cluster"] = [f"Cluster {l+1}" for l in labels]

        var = pca.explained_variance_ratio_
        return sub, var

    genres_key = tuple(sorted(selected_genres)) if selected_genres else None
    n_clusters = 5  # fixed default — slider removed from sidebar
    cluster_df, var_explained = compute_clusters(genres_key, pop_range[0], pop_range[1], n_clusters)

    st.caption(f"{len(cluster_df):,} tracks sampled")

    cluster_palette = px.colors.qualitative.Bold[:n_clusters]
    cluster_profile = cluster_df.groupby("Cluster")[AUDIO_FEATURES + ["popularity"]].mean()

    col_c1, col_c2 = st.columns([2, 3])

    with col_c1:
        # Average popularity by cluster
        cp_pop = (cluster_df.groupby("Cluster")["popularity"]
                  .mean().reset_index()
                  .sort_values("popularity", ascending=False))
        fig_cpp = px.bar(
            cp_pop, x="Cluster", y="popularity",
            color="Cluster",
            color_discrete_sequence=cluster_palette,
            template=PLOTLY_TEMPLATE, height=360,
            labels={"popularity": "Avg Popularity"},
        )
        fig_cpp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font=dict(color="#b3b3b3"),
            title=dict(text="Avg Popularity by Cluster", font=dict(size=13, color="#fff")),
            margin=dict(l=0, r=0, t=40, b=10),
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_cpp, use_container_width=True)

    with col_c2:
        # Audio fingerprint heatmap
        fig_cp = px.imshow(
            cluster_profile[AUDIO_FEATURES].T,
            color_continuous_scale=["#0d0d0d", "#1DB954"],
            aspect="auto",
            template=PLOTLY_TEMPLATE,
            zmin=0, zmax=1,
            labels=dict(color="Score"),
        )
        fig_cp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b3b3b3"),
            title=dict(text="Cluster Audio Fingerprints", font=dict(size=13, color="#fff")),
            margin=dict(l=0, r=0, t=40, b=10),
            height=360,
            coloraxis_colorbar=dict(tickfont=dict(color="#b3b3b3"), title="Norm"),
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Cluster Genre Composition</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Which genres dominate each audio cluster?</div>', unsafe_allow_html=True)

    cluster_genre_ct = (cluster_df.groupby(["Cluster", "track_genre"])
                        .size().reset_index(name="count"))
    fig_cg = px.bar(
        cluster_genre_ct, x="Cluster", y="count", color="track_genre",
        color_discrete_sequence=GENRE_COLOR_SEQ,
        template=PLOTLY_TEMPLATE, height=400,
        labels={"count": "Track Count", "track_genre": "Genre"},
        barmode="stack",
    )
    fig_cg.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b3b3b3"),
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
        legend=dict(bgcolor="rgba(18,18,18,0.95)", font=dict(size=12, color="#ffffff"),
                    title=dict(text="Genre", font=dict(size=13, color="#1DB954")),
                    itemsizing="constant", bordercolor="#333", borderwidth=1),
    )
    st.plotly_chart(fig_cg, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#555; font-size:12px;">'
    '<span style="color:#1DB954;">♪</span> Spotify Audio Landscape Dashboard · 113,842 tracks · 114 genres · Built with Streamlit + Plotly'
    '</p>',
    unsafe_allow_html=True,
)
