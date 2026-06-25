import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum · Customer Segmentation & Recommendations",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Theme & Navigation State Initializations
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Explicitly maintain the current page value
if "current_page" not in st.session_state:
    st.session_state.current_page = "📊 Overview"

# Theme palette
if st.session_state.dark_mode:
    T = {
        "app_bg":       "#0d1117",
        "sidebar_bg":   "#161b22",
        "card_bg":      "linear-gradient(135deg, #161b22 0%, #1c2128 100%)",
        "card_border":  "#30363d",
        "text_primary": "#e6edf3",
        "text_muted":   "#8b949e",
        "accent":       "#58a6ff",
        "hero_accent":  "#d2a8ff",
        "rec_card_bg":  "#161b22",
        "scrollbar_bg": "#0d1117",
        "scrollbar_thumb": "#30363d",
        "plot_paper":   "rgba(0,0,0,0)",
        "plot_grid":    "#30363d",
        "plot_grid2":   "#21262d",
        "legend_bg":    "rgba(22,27,34,0.8)",
        "input_bg":     "#161b22",
        "no_match_bg":  "#1c1f26",
        "cust_card_bg": "#161b22",
    }
else:
    T = {
        "app_bg":       "#f5f7fa",
        "sidebar_bg":   "#ffffff",
        "card_bg":      "linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%)",
        "card_border":  "#d0d7de",
        "text_primary": "#1f2328",
        "text_muted":   "#656d76",
        "accent":       "#0969da",
        "hero_accent":  "#6f42c1",
        "rec_card_bg":  "#ffffff",
        "scrollbar_bg": "#f5f7fa",
        "scrollbar_thumb": "#d0d7de",
        "plot_paper":   "rgba(0,0,0,0)",
        "plot_grid":    "#d0d7de",
        "plot_grid2":   "#e8ecf0",
        "legend_bg":    "rgba(255,255,255,0.9)",
        "input_bg":     "#ffffff",
        "no_match_bg":  "#fff5f5",
        "cust_card_bg": "#ffffff",
    }

plotly_font_color = T["text_primary"]

# ─────────────────────────────────────────────
# Custom CSS (theme-aware)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background: {T['app_bg']}; color: {T['text_primary']}; }}
    [data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; border-right: 1px solid {T['card_border']}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .stMarkdown, .stText, label, p {{ color: {T['text_primary']} !important; }}
    
    /* Hero Banner Component styling */
    .hero-banner {{
        background: {T['card_bg']};
        border: 1px solid {T['card_border']};
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }}
    .hero-tagline {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: {T['accent']};
        margin-bottom: 12px;
    }}
    .hero-title {{
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        color: {T['text_primary']};
        margin: 0 0 12px 0;
    }}
    .hero-highlight {{
        color: {T['hero_accent']};
    }}
    .hero-subtitle {{
        font-size: 14px;
        color: {T['text_muted']};
        max-width: 750px;
        line-height: 1.6;
        margin: 0;
    }}

    .metric-card {{
        background: {T['card_bg']};
        border: 1px solid {T['card_border']}; border-radius: 12px;
        padding: 20px 24px; transition: border-color 0.2s;
    }}
    .metric-card:hover {{ border-color: {T['accent']}; }}
    .metric-label {{ font-size:11px; font-weight:600; letter-spacing:0.1em;
        text-transform:uppercase; color:{T['text_muted']}; margin-bottom:6px; }}
    .metric-value {{ font-size:32px; font-weight:700; color:{T['text_primary']}; line-height:1.1; }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }}
    .section-header {{ font-size:18px; font-weight:700; color:{T['text_primary']}; margin:0 0 4px 0; }}
    .section-sub {{ font-size:13px; color:{T['text_muted']}; margin-bottom:20px; }}
    .rec-card {{
        background:{T['rec_card_bg']}; border:1px solid {T['card_border']};
        border-left:3px solid {T['accent']}; border-radius:10px;
        padding:14px 18px; margin:6px 0; display:flex; align-items:center; gap:12px;
    }}
    .rec-rank {{ font-size:22px; font-weight:700; color:{T['accent']}; min-width:32px; }}
    .rec-name {{ font-size:15px; font-weight:600; color:{T['text_primary']}; }}
    .predict-result {{ border-radius:12px; padding:28px; margin-top:16px; text-align:center; }}
    .predict-label {{ font-size:14px; text-transform:uppercase; letter-spacing:0.12em;
        color:{T['text_muted']}; margin-bottom:8px; }}
    .predict-segment {{ font-size:36px; font-weight:800; }}
    .predict-desc {{ font-size:14px; color:{T['text_muted']}; margin-top:10px; }}
    #MainMenu, footer {{ visibility:hidden; }}
    ::-webkit-scrollbar {{ width:6px; }}
    ::-webkit-scrollbar-track {{ background:{T['scrollbar_bg']}; }}
    ::-webkit-scrollbar-thumb {{ background:{T['scrollbar_thumb']}; border-radius:3px; }}
    
    /* INPUTS & MULTISELECT (Fixed too-dark backgrounds in light mode) */
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
        background:{T['input_bg']} !important; color:{T['text_primary']} !important;
        border-color:{T['card_border']} !important;
    }}
    [data-testid="stSelectbox"] > div, [data-testid="stMultiSelect"] > div {{
        background:{T['input_bg']} !important; color:{T['text_primary']} !important;
        border: 1px solid {T['card_border']};
    }}
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        background-color: {T['accent']}22 !important;
        border: 1px solid {T['accent']}40;
    }}
    [data-testid="stRadio"] label {{ color:{T['text_primary']} !important; }}
    [data-testid="stExpander"] {{ background:{T['rec_card_bg']} !important; border-color:{T['card_border']} !important; }}
    
    /* DATAFRAME & TABLES (Fixed harsh dark background) */
    [data-testid="stDataFrame"] {{ background:{T['rec_card_bg']} !important; }}
    div[data-testid="stDataFrame"] > div {{ background:{T['rec_card_bg']} !important; }}

    /* DOWNLOAD BUTTONS (Invert secondary button dark fills in light mode) */
    [data-testid="stDownloadButton"] button {{
        background-color: {T['input_bg']} !important;
        color: {T['text_primary']} !important;
        border: 1px solid {T['card_border']} !important;
    }}
    [data-testid="stDownloadButton"] button:hover {{
        border-color: {T['accent']} !important;
        color: {T['accent']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly Layout Global Configuration
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=T["plot_paper"], plot_bgcolor=T["plot_paper"],
    font=dict(color=plotly_font_color, family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor=T["legend_bg"], bordercolor=T["card_border"], borderwidth=1, font=dict(color=T["text_primary"])),
)

# ─────────────────────────────────────────────
# Segment config
# ─────────────────────────────────────────────
SEGMENT_CONFIG = {
    "High-Value":  {"color": "#d2a8ff", "icon": "🏆", "bg": "rgba(210,168,255,0.12)",
                    "desc": "Recent, frequent, and high-spending customers. Prioritize retention and VIP offers."},
    "Regular":     {"color": "#58a6ff", "icon": "💎", "bg": "rgba(88,166,255,0.12)",
                    "desc": "Steady purchasers with moderate spend. Nurture with loyalty programs."},
    "Occasional":  {"color": "#3fb950", "icon": "⚡", "bg": "rgba(63,185,80,0.12)",
                    "desc": "Infrequent buyers with lower spend. Win-back with targeted discounts."},
    "At-Risk":     {"color": "#f85149", "icon": "⚠️", "bg": "rgba(248,81,73,0.12)",
                    "desc": "Haven't purchased in a long time. Urgent re-engagement needed."},
    "High Value Customers": {"color": "#d2a8ff", "icon": "🏆", "bg": "rgba(210,168,255,0.12)",
                             "desc": "High value customers — prioritize retention."},
    "Loyal Customers":      {"color": "#58a6ff", "icon": "💎", "bg": "rgba(88,166,255,0.12)",
                             "desc": "Loyal repeat buyers — nurture with rewards."},
    "Active Customers":     {"color": "#3fb950", "icon": "⚡", "bg": "rgba(63,185,80,0.12)",
                             "desc": "Recently active customers — keep them engaged."},
    "At Risk Customers":    {"color": "#f85149", "icon": "⚠️", "bg": "rgba(248,81,73,0.12)",
                             "desc": "At-risk customers — re-engage urgently."},
}

def hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def metric_card(label, value):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{T['text_primary']}">{value}</div>
    </div>"""

def render_hero_banner(tagline, title_left, title_highlight, title_right, subtitle):
    """Renders a structured, polished interactive header container."""
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-tagline">{tagline}</div>
        <h1 class="hero-title">{title_left} <span class="hero-highlight">{title_highlight}</span> {title_right}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data & Model Loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_segments():
    for path in ["data/customer_segments.csv", "customer_segments.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Segment"] = df["Segment"].str.strip()
            return df
    st.error("⚠️  `data/customer_segments.csv` not found. Run the notebook first.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_kmeans_model():
    for path in ["data/kmeans_model.pkl", "kmeans_model.pkl"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_similarity_matrix():
    sim_path, prod_path = None, None
    for p in ["data/similarity_matrix.pkl", "similarity_matrix.pkl"]:
        if os.path.exists(p):
            sim_path = p
    for p in ["data/product_index.pkl", "product_index.pkl"]:
        if os.path.exists(p):
            prod_path = p
    if sim_path and prod_path:
        with open(sim_path, "rb") as f:
            sim = pickle.load(f)
        with open(prod_path, "rb") as f:
            prod_idx = pickle.load(f)
        return sim, prod_idx
    return None, None

@st.cache_data(show_spinner=False)
def compute_summary(df):
    return (
        df.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
        .mean().round(2).reset_index()
    )

def get_recommendations(product_name, sim_matrix, product_index, n=5):
    lower_map = {k.lower(): k for k in product_index}
    query = product_name.strip().lower()
    if query in lower_map:
        actual_key = lower_map[query]
    else:
        matches = [k for k in lower_map if query in k]
        if not matches:
            return None, []
        actual_key = lower_map[matches[0]]
    idx = product_index[actual_key]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)
    top = [(i, s) for i, s in scores if i != idx][:n]
    inv_index = {v: k for k, v in product_index.items()}
    results = [(inv_index[i], round(s, 4)) for i, s in top]
    return actual_key, results

def predict_cluster(recency, frequency, monetary, model_bundle, df_segments):
    if model_bundle:
        scaler   = model_bundle["scaler"]
        kmeans   = model_bundle["kmeans"]
        label_map = model_bundle.get("label_map", {})
        X = scaler.transform([[recency, frequency, monetary]])
        cluster_id = int(kmeans.predict(X)[0])
        return label_map.get(cluster_id, f"Cluster {cluster_id}")
    else:
        summary = compute_summary(df_segments)
        best_seg, best_dist = None, float("inf")
        r_range = df_segments["Recency"].max()  - df_segments["Recency"].min()  + 1e-9
        f_range = df_segments["Frequency"].max() - df_segments["Frequency"].min() + 1e-9
        m_range = df_segments["Monetary"].max()  - df_segments["Monetary"].min()  + 1e-9
        for _, row in summary.iterrows():
            dist = (
                ((recency   - row["Recency"])   / r_range) ** 2 +
                ((frequency - row["Frequency"]) / f_range) ** 2 +
                ((monetary  - row["Monetary"])  / m_range) ** 2
            ) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_seg = row["Segment"]
        return best_seg

# ─────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────
df_raw       = load_segments()
model_bundle = load_kmeans_model()
sim_matrix, product_index = load_similarity_matrix()
segments = sorted(df_raw["Segment"].unique().tolist())

# ─────────────────────────────────────────────
# Sidebar & Persistent Page Switching Logic
# ─────────────────────────────────────────────
with st.sidebar:
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown(f"## 🛒 Shopper Spectrum")
    with col_toggle:
        st.markdown("<div style='margin-top:14px'>", unsafe_allow_html=True)
        toggle_label = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(toggle_label, help="Toggle light/dark mode", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{T['card_border']};margin:8px 0 16px'>", unsafe_allow_html=True)

    page_options = [
        "Overview", 
        "Product Recommendations", 
        "Predict Customer Segment",
        "Segment Deep-Dive", 
        "Customer Lookup", 
        "Export"
    ]
    
    try:
        current_page_idx = page_options.index(st.session_state.current_page)
    except ValueError:
        current_page_idx = 0

    page = st.radio(
        "Navigate",
        page_options,
        index=current_page_idx,
        label_visibility="collapsed",
    )
    st.session_state.current_page = page

    st.markdown(f"<hr style='border-color:{T['card_border']};margin:16px 0 12px'>", unsafe_allow_html=True)
    selected_segments = st.multiselect("Filter segments", options=segments, default=segments)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("K-Means · Collaborative Filtering · RFM")

df      = df_raw[df_raw["Segment"].isin(selected_segments)]
summary = compute_summary(df)

# ══════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════
if page == "Overview":
    render_hero_banner(
        tagline="Powered by ML · RFM Clustering · SciKit-Learn",
        title_left="Understand",
        title_highlight="Customer Spectrums",
        title_right="Before Intent Shifts",
        subtitle="Advanced Recency, Frequency, and Monetary cohort extraction maps automated purchasing fingerprints into highly descriptive customer lifecycles instantly."
    )

    total = len(df)
    avg_r = round(df["Recency"].mean(), 1)
    avg_f = round(df["Frequency"].mean(), 2)
    avg_m = round(df["Monetary"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Total Customers", f"{total:,}"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Avg Recency (days)", f"{avg_r}"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Avg Frequency", f"{avg_f}"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("Avg Monetary", f"${avg_m:,.2f}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown('<p class="section-header">Segment Distribution</p><p class="section-sub">Share of customers per segment</p>', unsafe_allow_html=True)
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        colors = [SEGMENT_CONFIG.get(s, {}).get("color", "#58a6ff") for s in seg_counts["Segment"]]
        fig_donut = go.Figure(go.Pie(
            labels=seg_counts["Segment"], values=seg_counts["Count"], hole=0.62,
            marker=dict(colors=colors, line=dict(color=T["app_bg"], width=2)),
            textinfo="percent", textfont=dict(size=12, color=T["text_primary"]), 
            hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{total:,}</b><br><span style='font-size:11px'>customers</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color=T["text_primary"]), align="center",
        )
        fig_donut.update_layout(**{**PLOTLY_LAYOUT, "height": 320, "showlegend": True,
                                   "legend": dict(orientation="v", x=1, y=0.5,
                                   bgcolor=T["legend_bg"], bordercolor=T["card_border"], borderwidth=1, font=dict(color=T["text_primary"]))})
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown('<p class="section-header">RFM Comparison by Segment</p><p class="section-sub">Mean values across Recency, Frequency, Monetary</p>', unsafe_allow_html=True)
        metrics_list = ["Recency", "Frequency", "Monetary"]
        fig_radar = go.Figure()
        for _, row in summary.iterrows():
            seg = row["Segment"]
            cfg = SEGMENT_CONFIG.get(seg, {"color": "#58a6ff", "icon": ""})
            vals = [row[m] for m in metrics_list] + [row[metrics_list[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=metrics_list + [metrics_list[0]],
                fill="toself", name=f"{cfg['icon']} {seg}",
                line=dict(color=cfg["color"], width=2),
                fillcolor=hex_to_rgba(cfg["color"], 0.15), opacity=0.85,
            ))
        fig_radar.update_layout(**{**PLOTLY_LAYOUT, "height": 320,
                                   "polar": dict(bgcolor="rgba(0,0,0,0)",
                                   radialaxis=dict(visible=True, gridcolor=T["plot_grid"], color=T["text_primary"]),
                                   angularaxis=dict(gridcolor=T["plot_grid"], color=T["text_primary"]))})
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="section-header">Average RFM Metrics by Segment</p>', unsafe_allow_html=True)
    fig_bar = make_subplots(rows=1, cols=3, subplot_titles=["Recency (days)", "Frequency (orders)", "Monetary ($)"])
    for i, metric in enumerate(["Recency", "Frequency", "Monetary"], start=1):
        for _, row in summary.iterrows():
            seg = row["Segment"]
            cfg = SEGMENT_CONFIG.get(seg, {"color": "#58a6ff"})
            fig_bar.add_trace(
                go.Bar(x=[seg], y=[row[metric]], name=seg, showlegend=(i == 1),
                       marker_color=cfg["color"],
                       textfont=dict(color=T["text_primary"]),
                       hovertemplate=f"<b>{seg}</b><br>{metric}: %{{y}}<extra></extra>"),
                row=1, col=i)
    fig_bar.update_layout(**{**PLOTLY_LAYOUT, "height": 320, "barmode": "group",
                              "xaxis":  dict(showticklabels=False, gridcolor=T["plot_grid"]),
                              "xaxis2": dict(showticklabels=False, gridcolor=T["plot_grid"]),
                              "xaxis3": dict(showticklabels=False, gridcolor=T["plot_grid"]),
                              "yaxis":  dict(gridcolor=T["plot_grid2"], tickfont=dict(color=T["text_primary"])),
                              "yaxis2": dict(gridcolor=T["plot_grid2"], tickfont=dict(color=T["text_primary"])),
                              "yaxis3": dict(gridcolor=T["plot_grid2"], tickfont=dict(color=T["text_primary"]))})
    
    for annotation in fig_bar['layout']['annotations']:
        annotation['font'] = dict(color=T["text_primary"], size=14)
        
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════
# PAGE: 🎯 Product Recommendations
# ══════════════════════════════════════════════
elif page == "Product Recommendations":
    render_hero_banner(
        tagline="COLLABORATIVE FILTERING · COSINE SIMILARITY",
        title_left="Discover",
        title_highlight="Purchase Affinities",
        title_right="Instantly",
        subtitle="Unveil matrix-factorized relations across your store inventory categories. Provide accurate items based directly on transactional correlation scores."
    )

    if sim_matrix is None or product_index is None:
        st.warning("⚠️ **Similarity matrix not found.**")
    else:
        all_products = sorted(product_index.keys())

        col_sel, col_btn = st.columns([4, 1])
        with col_sel:
            product_input = st.selectbox(
                "Select a Product Category",
                options=[""] + all_products,
                format_func=lambda x: "— Choose a category —" if x == "" else x.replace("_", " ").title(),
            )
        with col_btn:
            st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
            btn_rec = st.button("Get", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption(f"📦 {len(all_products)} product categories available")

        if product_input:
            matched_name, recs = get_recommendations(product_input, sim_matrix, product_index, n=5)
            if matched_name is None:
                st.error(f'❌ No product matching "{product_input}" found.')
            else:
                st.markdown(f'<p style="color:#3fb950;font-size:13px;margin:12px 0 8px">✅ Showing recommendations for: <b>{matched_name.replace("_", " ").title()}</b></p>', unsafe_allow_html=True)
                st.markdown('<p class="section-header" style="margin-top:16px">Top 5 Similar Product Categories</p>', unsafe_allow_html=True)

                icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for rank, (name, score) in enumerate(recs, start=1):
                    bar_width = max(4, int(score * 100))
                    display_name = name.replace("_", " ").title()
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-rank">{icons[rank-1]}</div>
                        <div style="flex:1">
                            <div class="rec-name">{display_name}</div>
                            <div style="font-size:11px;color:{T["text_muted"]};margin-top:2px">{name}</div>
                            <div style="margin-top:6px;display:flex;align-items:center;gap:10px">
                                <div style="flex:1;height:6px;background:{T["plot_grid2"]};border-radius:3px;overflow:hidden">
                                    <div style="width:{bar_width}%;height:100%;background:{T["accent"]};border-radius:3px"></div>
                                </div>
                                <div style="font-size:12px;color:{T["text_muted"]};min-width:90px">Similarity: {score:.4f}</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: 🔮 Predict Customer Segment
# ══════════════════════════════════════════════
elif page == "Predict Customer Segment":
    render_hero_banner(
        tagline="K-MEANS CLUSTERING · VECTOR ASSIGNMENT",
        title_left="Forecast",
        title_highlight="User Classifications",
        title_right="in Real-Time",
        subtitle="Pass customized client properties into the normalized coordinate vector space to allocate live leads into precise operational funnels."
    )

    if model_bundle is None:
        st.info("ℹ️ **No saved model found** — using centroid-distance fallback from existing segment data.")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=3000, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=1000, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary (total spend, $)", min_value=0.0, max_value=100000.0, value=250.0, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("Predict Cluster", type="primary")

    if btn_predict:
        predicted_segment = predict_cluster(recency, frequency, monetary, model_bundle, df_raw)
        cfg = SEGMENT_CONFIG.get(predicted_segment, {
            "color": "#58a6ff", "icon": "🔵", "bg": "rgba(88,166,255,0.12)", "desc": "Customer segment predicted."
        })

        st.markdown(f"""
        <div class="predict-result" style="background:{cfg['bg']};border:1px solid {cfg['color']}40">
            <div class="predict-label">Predicted Segment</div>
            <div class="predict-segment" style="color:{cfg['color']}">{cfg['icon']} {predicted_segment}</div>
            <div class="predict-desc">{cfg['desc']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Your Input vs Segment Averages</p>', unsafe_allow_html=True)

        comp_df = compute_summary(df_raw).copy()
        your_row = pd.DataFrame([{"Segment": "⭐ Your Input", "Recency": recency, "Frequency": frequency, "Monetary": monetary}])
        comp_df = pd.concat([comp_df, your_row], ignore_index=True)

        for metric, label in [("Recency", "Recency (days)"), ("Frequency", "# Orders"), ("Monetary", "Spend ($)")]:
            bar_colors = ["#f85149" if s == "⭐ Your Input" else SEGMENT_CONFIG.get(s, {}).get("color", "#58a6ff") for s in comp_df["Segment"]]
            fig = go.Figure(go.Bar(
                x=comp_df["Segment"], y=comp_df[metric],
                marker_color=bar_colors,
                text=comp_df[metric].round(1),
                textposition="outside",
                textfont=dict(color=T["text_primary"])
            ))
            fig.update_layout(**{**PLOTLY_LAYOUT, "height": 260, "title": label,
                                 "xaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"])),
                                 "yaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]))})
            fig.update_layout(title_font_color=T["text_primary"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════
# PAGE: 🔍 Segment Deep-Dive
# ══════════════════════════════════════════════
elif page == "Segment Deep-Dive":
    render_hero_banner(
        tagline="COHORT ISOLATION · DENSITY ANALYSIS",
        title_left="Deconstruct",
        title_highlight="Target Cohorts",
        title_right="Granularly",
        subtitle="Isolate metric distributions and check variances inside specific groups to understand customer spending habits."
    )

    # Added a blank placeholder to options, matching the look and feel of the recommendation page
    chosen = st.selectbox(
        "Select a segment to inspect", 
        options=[""] + segments,
        format_func=lambda s: "— Choose a segment —" if s == "" else f"{SEGMENT_CONFIG.get(s, {}).get('icon', '🔵')}  {s}"
    )

    # Only render the metrics and charts if a valid segment has been chosen
    if chosen == "":
        st.info("💡 **Please select a customer segment from the dropdown menu above to inspect its metrics.**")
    else:
        seg_df = df[df["Segment"] == chosen]
        cfg = SEGMENT_CONFIG.get(chosen, {"color": "#58a6ff", "icon": "🔵", "bg": "rgba(88,166,255,0.1)"})

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("Customers in segment", f"{len(seg_df):,}"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Avg Recency", f"{seg_df['Recency'].mean():.1f} days"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Avg Frequency", f"{seg_df['Frequency'].mean():.2f}"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Avg Monetary", f"${seg_df['Monetary'].mean():,.2f}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            fig_r = px.histogram(seg_df, x="Recency", nbins=40, color_discrete_sequence=[cfg["color"]], title="Recency Distribution")
            fig_r.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                   "xaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]), title=dict(font=dict(color=T["text_primary"]))), 
                                   "yaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]), title=dict(font=dict(color=T["text_primary"])))})
            fig_r.update_layout(title_font_color=T["text_primary"])
            fig_r.update_traces(marker_line_color=T["app_bg"], marker_line_width=0.5)
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})
        with col_r:
            fig_m = px.histogram(seg_df, x="Monetary", nbins=40, color_discrete_sequence=[cfg["color"]], title="Monetary Distribution")
            fig_m.update_layout(**{**PLOTLY_LAYOUT, "height": 280,
                                   "xaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]), title=dict(font=dict(color=T["text_primary"]))), 
                                   "yaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]), title=dict(font=dict(color=T["text_primary"])))})
            fig_m.update_layout(title_font_color=T["text_primary"])
            fig_m.update_traces(marker_line_color=T["app_bg"], marker_line_width=0.5)
            st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})

        fig_scatter = px.scatter(
            seg_df.sample(min(2000, len(seg_df)), random_state=42),
            x="Recency", y="Monetary",
            color_discrete_sequence=[cfg["color"]], opacity=0.5,
            title="Recency vs Monetary (sampled)",
            labels={"Recency": "Recency (days)", "Monetary": "Monetary ($)"},
        )
        fig_scatter.update_layout(**{**PLOTLY_LAYOUT, "height": 300,
                                     "xaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]), title=dict(font=dict(color=T["text_primary"]))), 
                                     "yaxis": dict(gridcolor=T["plot_grid"], tickfont=dict(color=T["text_primary"]), title=dict(font=dict(color=T["text_primary"])))})
        fig_scatter.update_layout(title_font_color=T["text_primary"])
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

        with st.expander("📋 Raw segment data (first 200 rows)"):
            st.dataframe(seg_df.head(200).reset_index(), use_container_width=True, hide_index=True)
# ══════════════════════════════════════════════
# PAGE: Customer Lookup
# ══════════════════════════════════════════════
elif page == "Customer Lookup":
    render_hero_banner(
        tagline="INDEX INDEXATION · INDIVIDUAL METRICS",
        title_left="Audit",
        title_highlight="User Accounts",
        title_right="Instantly",
        subtitle="Retrieve specific historical records, check their lifetime value status, and view exact metrics instantly."
    )

    id_col = None
    for col in ["customer_unique_id", "CustomerID", "customer_id"]:
        if col in df_raw.columns:
            id_col = col
            break

    if id_col is None:
        st.error("No customer ID column found in the dataset.")
    else:
        all_ids = sorted(df_raw[id_col].astype(str).unique().tolist())
        st.caption(f"🔍 {len(all_ids):,} customers in dataset — type to search")
        query = st.selectbox(
            "Customer ID",
            options=[""] + all_ids,
            format_func=lambda x: "— Select a customer ID —" if x == "" else x,
            label_visibility="collapsed",
        )
        if query:
            result = df_raw[df_raw[id_col].astype(str) == query]
            if len(result) > 0:
                row = result.iloc[0]
                seg = row.get("Segment", "Unknown")
                cfg = SEGMENT_CONFIG.get(seg, {"color": "#58a6ff", "icon": "🔵", "bg": "rgba(88,166,255,0.1)"})
                badge = f'<span class="badge" style="background:{cfg["bg"]};color:{cfg["color"]}">{cfg["icon"]} {seg}</span>'
                st.markdown(f"""
                <div style="background:{T['rec_card_bg']};border:1px solid {T['card_border']};border-left:3px solid #58a6ff;border-radius:10px;padding:20px;margin-top:12px">
                    <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:16px">
                        <div>
                            <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;color:{T["text_muted"]};margin-bottom:4px">Customer ID</div>
                            <div style="font-family:monospace;font-size:13px;color:#58a6ff;background:{T['app_bg']};padding:4px 8px;border-radius:6px">{row[id_col]}</div>
                        </div>
                        <div>{badge}</div>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
                        <div><div style="font-size:11px;text-transform:uppercase;color:{T["text_muted"]};margin-bottom:4px">Recency</div>
                        <div style="font-size:24px;font-weight:700;color:{T["text_primary"]}">{int(row['Recency'])} <span style="font-size:13px;color:{T["text_muted"]}">days</span></div></div>
                        <div><div style="font-size:11px;text-transform:uppercase;color:{T["text_muted"]};margin-bottom:4px">Frequency</div>
                        <div style="font-size:24px;font-weight:700;color:{T["text_primary"]}">{int(row['Frequency'])} <span style="font-size:13px;color:{T["text_muted"]}">orders</span></div></div>
                        <div><div style="font-size:11px;text-transform:uppercase;color:{T["text_muted"]};margin-bottom:4px">Monetary</div>
                        <div style="font-size:24px;font-weight:700;color:{T["text_primary"]}">${float(row['Monetary']):,.2f}</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:{T['no_match_bg']};border:1px solid #f85149;border-radius:10px;padding:20px;margin-top:12px;color:#f85149">
                    ❌ No customer found with that ID.
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: Export
# ══════════════════════════════════════════════
elif page == "Export":
    render_hero_banner(
        tagline="PIPELINE COMPILATION · FILE GENERATION",
        title_left="Extract",
        title_highlight="Structured Datasets",
        title_right="Seamlessly",
        subtitle="Export localized cohort slices or full tabular evaluations directly to universal formats for down-stream activation campaigns."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Filtered data** (based on sidebar selection)")
        st.caption(f"{len(df):,} rows · {', '.join(selected_segments)}")
        st.download_button("⬇ Download filtered CSV", df.to_csv(index=False).encode("utf-8"),
                           "customer_segments_filtered.csv", "text/csv", use_container_width=True)
    with col2:
        st.markdown("**Full dataset**")
        st.caption(f"{len(df_raw):,} rows · all segments")
        st.download_button("⬇ Download full CSV", df_raw.to_csv(index=False).encode("utf-8"),
                           "customer_segments_full.csv", "text/csv", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Segment summary table**")
    st.dataframe(compute_summary(df_raw), use_container_width=True, hide_index=True)