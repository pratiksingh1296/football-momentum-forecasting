# ==== Imports ====

# Standard library
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party
import streamlit as st
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
import plotly.express as px
import plotly.graph_objects as go

# Local modules
from transformers import AutoTokenizer
from src.model import DistilBertMomentum
from config import BASE_DIR, NUM_CLASSES, DROPOUT, CLASS_NAMES

st.set_page_config(page_title="Football Momentum Forecasting", layout="wide")

# ===== Cached Resources ====

# ---- Model loading ----
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device('cpu')  # deployment target likely has no GPU; keep it simple and portable

    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR / 'models' / 'tokenizer')

    model = DistilBertMomentum(num_classes=NUM_CLASSES, dropout=DROPOUT)

    checkpoint_path = hf_hub_download(
        repo_id="leoken/football-momentum-distilbert",
        filename="bert_momentum.pth"
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# ---- Curated match data ----
@st.cache_data
def load_curated_matches():
    test_df = pd.read_csv(BASE_DIR / 'data' / 'exp_a_prefix_test_sample.csv')
    return test_df

matches_df = load_curated_matches()
CURATED_MATCH_IDS = matches_df['id_odsp'].unique().tolist()

# ---- Load Raw events ----
@st.cache_data
def load_raw_events():
    return pd.read_csv(BASE_DIR / 'data' / 'curated_raw_events.csv')  

raw_events_df = load_raw_events()

# ==== Helper Functions ====

# ---- Prediction helper ----
def predict_window(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    return probs

# cache for quicker button response
@st.cache_data
def predict_window_cached(text):
    return predict_window(text)

# ---- Momentum meter ------
def momentum_meter_value(probs):
    # probs[0] = P(Home Dominant), probs[1] = P(Away Dominant)
    return probs[0] - probs[1]  # ranges roughly -1 (Away) to +1 (Home)


# ==== UI ==== 

st.set_page_config(
    page_title="Match Replay",
    layout="wide"
)

st.title("⚽ Football Momentum Forecasting")

st.markdown("""
A DistilBERT model fine-tuned to classify football commentary in 5-minute windows
as **Home Dominant**, **Away Dominant**, or **Balanced**. It achieves a 91% macro F1
score on held-out matches, improving on a 72% LSTM baseline. Select a match below
and step through it window by window to explore the model's predictions alongside
the match's actual momentum.
""")

match_labels = {}
for match_id in CURATED_MATCH_IDS:
    match_events = raw_events_df[raw_events_df['id_odsp'] == match_id]
    home_team = match_events[match_events['side'] == 1]['event_team'].iloc[0]
    away_team = match_events[match_events['side'] == 2]['event_team'].iloc[0]
    match_labels[f"{home_team} vs {away_team}"] = match_id

st.markdown("### Choose a match")
selected_label = st.selectbox("", list(match_labels.keys()))
selected_match = match_labels[selected_label]

match_windows = matches_df[matches_df['id_odsp'] == selected_match].sort_values('window_start').reset_index(drop=True)

# Reset the position if match is changed
if 'current_match' not in st.session_state or st.session_state.current_match != selected_match:
    st.session_state.current_match = selected_match
    st.session_state.window_idx = 0

# buttons
_, prev_col, reset_col, next_col, _ = st.columns([2, 1, 1, 1, 2])

with prev_col:
    if st.button("◀ Previous Window", disabled=st.session_state.window_idx == 0, use_container_width=True):
        if st.session_state.window_idx > 0:
            st.session_state.window_idx -= 1

with reset_col:
    if st.button("⏮ Reset", use_container_width=True):
        st.session_state.window_idx = 0

with next_col:
    if st.button("▶ Next Window", disabled=st.session_state.window_idx >= len(match_windows)-1, use_container_width=True): 
        if st.session_state.window_idx < len(match_windows) - 1:
            st.session_state.window_idx += 1

# calculate things based on updated index
current_row = match_windows.iloc[st.session_state.window_idx]
current_time = current_row["window_end"] 

# Events for this match up to the current time
events_so_far = raw_events_df[(raw_events_df["id_odsp"] == selected_match) &(raw_events_df["time"] <= current_time)]

# Count goals
home_score = events_so_far[(events_so_far["side"] == 1) &(events_so_far["is_goal"] == 1)].shape[0]
away_score = events_so_far[(events_so_far["side"] == 2) &(events_so_far["is_goal"] == 1)].shape[0]

home_team = match_events[match_events['side'] == 1]['event_team'].iloc[0]
away_team = match_events[match_events['side'] == 2]['event_team'].iloc[0]

# show score-board
with st.container(border=True):
    st.markdown(
        f"""
        <h1 style="text-align:center">
        {home_team} {home_score} - {away_score} {away_team}
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.caption(f"Minute {current_time}")

# show commentary
st.subheader(f"Window {st.session_state.window_idx + 1} / {len(match_windows)}  (minute {current_row['window_start']}–{current_row['window_end']})")
with st.expander("Window Commentary", expanded=True):
    st.write(current_row["text"])

# calculate values
probs = predict_window_cached(current_row["text"])
pred_class = CLASS_NAMES[probs.index(max(probs))]
meter_value = momentum_meter_value(probs)

# window update
windows_so_far = match_windows.iloc[:st.session_state.window_idx + 1]
raw_momentum = windows_so_far['momentum_score'].tolist()
full_match_momentum = match_windows['momentum_score']
match_max_abs = max(full_match_momentum.abs().max(), 1)

momentum_history_normalized = [m / match_max_abs for m in raw_momentum]

meter_history = []
for _, row in windows_so_far.iterrows():
    p = predict_window_cached(row["text"])
    meter_history.append(momentum_meter_value(p))

#debug
debug = st.sidebar.toggle("Debug mode", value=False)
if debug:
    with st.expander("Debug: raw probabilities"):
        st.write({name: f"{p:.4f}" for name, p in zip(CLASS_NAMES, probs)})


# ==== Graphs ====

col1, col2 = st.columns([1,1])

# ---- Bar Chart ----
with col1:
    # Streamlit metric value styling
    # targets: st.metric(label, value, delta)
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 24px !important;   /* Change to desired size */
        }
        </style>
        """, unsafe_allow_html=True
        )
    st.markdown("### Prediction")
    st.metric(
        "",
        pred_class,
        f"{max(probs):.1%}"
    )

    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs
    })

    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        color="Probability",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---- Meter Gauge ----
with col2:
    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=meter_value,
    title={"text": "<b>Momentum Meter</b>", "font": {"size": 28}},
    number={"valueformat": ".2f"},
    gauge={
        "axis": {"range": [-1, 1]},
        "bar": {"color": "royalblue"},
        "steps": [
            {"range": [-1, 0], "color": "#ffb3b3"},
            {"range": [0, 1], "color": "#b3ffb3"},
            ],
        }
    ))

    fig.update_layout(height=350)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 20px;
        font-weight: 500;
        margin-top: -5px;
    ">
        Away ◀────────▶ Home
    </div>
    """,
    unsafe_allow_html=True,
    )

# --- Line Chart - Momentum ( -1 to 1 )

# show momentum line chart
st.subheader("Match Momentum")
st.markdown(
    "<span style='color:#8fdb8f'>■</span> Home Dominant &nbsp;&nbsp; "
    "<span style='color:#ffb3b3'>■</span> Away Dominant",
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_hrect(y0=0, y1=1, fillcolor="#b3ffb3", opacity=0.15, line_width=0)
fig.add_hrect(y0=-1, y1=0, fillcolor="#ffb3b3", opacity=0.15, line_width=0)

fig.add_hline(
    y=0,
    line_color="white",
    line_width=1,
    opacity=0.5
)

fig.add_trace(go.Scatter(
    x=list(range(1, len(momentum_history_normalized) + 1)),
    y=momentum_history_normalized,
    mode="lines+markers",
    line=dict(color="royalblue", width=3),
    name="Actual Momentum (Normalized)"
))

fig.add_trace(go.Scatter(
    x=list(range(1, len(meter_history) + 1)),
    y=meter_history,
    mode="lines+markers",
    line=dict(color="orange", width=2, dash="dot"),
    opacity=0.7,
    name="Model prediction"
))

fig.update_layout(
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    yaxis=dict(
        range=[-1, 1],
        title="Momentum"
    ),
    xaxis_title="Window",
    showlegend=True,
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Note: Model predictions are derived from softmax probabilities, "
    "which can become saturated near extreme values. "
    "Observed momentum typically changes more gradually."
)




