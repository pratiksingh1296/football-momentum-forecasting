import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from src.model import DistilBertMomentum
from config import BASE_DIR, NUM_CLASSES, DROPOUT, CLASS_NAMES

st.set_page_config(page_title="Try the Model", layout="wide")

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR / 'models' / 'tokenizer')
    model = DistilBertMomentum(num_classes=NUM_CLASSES, dropout=DROPOUT)
    checkpoint_path = hf_hub_download(repo_id="leoken/football-momentum-distilbert", filename="bert_momentum.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

def predict(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    return probs

st.title("🔍 Explore the Model")
st.markdown("See how the model handles real examples from its test set, including cases where it got it right, and cases where it didn't.")

examples_df = pd.read_csv(BASE_DIR / 'data' / 'curated_examples.csv')

tab1, tab2 = st.tabs(["Real Examples", "Try Your Own"])

with tab1:
    outcome_filter = st.radio("Show", ["All", "Correct predictions", "Misclassified"], horizontal=True)

    filtered = examples_df.copy()
    if outcome_filter == "Correct predictions":
        filtered = filtered[filtered['label'] == filtered['pred']]
    elif outcome_filter == "Misclassified":
        filtered = filtered[filtered['label'] != filtered['pred']]

    selected_idx = st.selectbox(
        "Pick an example",
        filtered.index,
        format_func=lambda i: filtered.loc[i, 'text'][:80] + "..."
    )
    row = filtered.loc[selected_idx]

    with st.expander("Commentary", expanded=True):
        st.write(row['text'])

    probs = predict(row['text'])
    pred_class = CLASS_NAMES[probs.index(max(probs))]
    true_class = CLASS_NAMES[int(row['label'])]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("True label", true_class)
    with col2:
        is_correct = pred_class == true_class
        st.metric("Model prediction", pred_class, "✓ Correct" if is_correct else "✗ Incorrect")

    st.bar_chart(pd.Series(probs, index=CLASS_NAMES))

with tab2:
    st.markdown("Describe an event, and pick which side it's about, the model reads text tagged this way during training.")

    side = st.radio("Which side is this event about?", ["Home", "Away"], horizontal=True)
    user_text = st.text_area("Describe the event", placeholder="e.g. Ronaldo takes a left-footed shot, narrowly missing the right post.")

    if st.button("Predict") and user_text.strip():
        tag = "HOME_TEAM" if side == "Home" else "AWAY_TEAM"
        tagged_text = f"[{tag}] {user_text.strip()}"

        st.caption(f"Model input: `{tagged_text}`")

        probs = predict(tagged_text)
        pred_class = CLASS_NAMES[probs.index(max(probs))]

        st.metric("Prediction", pred_class, f"{max(probs):.1%}")
        st.bar_chart(pd.Series(probs, index=CLASS_NAMES))