"""
Streamlit web application for Resume NER.

Launch with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import spacy
from spacy import displacy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
sys.path.insert(0, ROOT_DIR)

from src.predict import predict, load_model
from src.entity_colors import ENTITY_COLORS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume NER — Entity Extractor",
    page_icon="📄",
    layout="wide",
)
 
# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { padding-top: 2rem; }
    .stTextArea textarea { font-family: 'Courier New', monospace; font-size: 14px; }
    h1 { color: #2C3E50; }
    .entity-table { margin-top: 1rem; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 Resume NER — Named Entity Recognition")
st.markdown(
    "Paste a student resume below and click **Analyze** to extract entities "
    "such as **Name**, **Skills**, **Education**, **Company**, and more."
)

# ---------------------------------------------------------------------------
# Sidebar — instructions & label legend
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏷️ Entity Labels")
    for label, color in ENTITY_COLORS.items():
        st.markdown(
            f'<span style="color:#000;background-color:{color};padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;">'
            f"{label}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        "**How to use**\n"
        "1. Paste a resume in the text box\n"
        "2. Click **🔍 Analyze Resume**\n"
        "3. View highlighted entities & summary table"
    )

# ---------------------------------------------------------------------------
# Sample resume for quick demo
# ---------------------------------------------------------------------------
SAMPLE = (
    "Ravi Shankar\n"
    "Email: ravi.shankar@gmail.com\n"
    "Phone: 9123456780\n"
    "Education: B.Tech in Computer Science from IIT Bombay, 2023\n"
    "Skills: Python, Machine Learning, NLP, TensorFlow\n"
    "Experience: AI Engineer at Microsoft, Hyderabad"
)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    resume_text = st.text_area(
        "📝 Paste Resume Text",
        value=SAMPLE,
        height=300,
        help="Enter the raw text of a student resume",
    )
    analyze_btn = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load and cache the trained NER model."""
    return load_model(MODEL_DIR)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_btn:
    if not resume_text.strip():
        st.warning("Please enter some resume text first.")
    elif not os.path.isdir(MODEL_DIR):
        st.error(
            "❌ Trained model not found! Run the following command first:\n\n"
            "```\npython src/train.py\n```"
        )
    else:
        nlp = get_model()
        entities = predict(resume_text, nlp)

        with col2:
            # ---------- Highlighted entity visualization ----------
            st.subheader("🎯 Detected Entities")
            doc = nlp(resume_text)

            # displacy options
            options = {
                "ents": list(ENTITY_COLORS.keys()),
                "colors": ENTITY_COLORS,
            }
            html = displacy.render(doc, style="ent", options=options)
            st.markdown(
                f'<div style="background:#FAFAFA;color:#333;padding:16px;border-radius:8px;'
                f'line-height:2.2;font-size:15px;">{html}</div>',
                unsafe_allow_html=True,
            )

        # ---------- Summary table ----------
        st.markdown("---")
        st.subheader("📊 Entity Summary Table")

        if entities:
            # Group by label
            from collections import defaultdict
            grouped = defaultdict(list)
            for ent in entities:
                grouped[ent["label"]].append(ent["text"])

            table_data = []
            for label in ENTITY_COLORS:
                if label in grouped:
                    table_data.append(
                        {
                            "Label": label,
                            "Entities Found": ", ".join(grouped[label]),
                            "Count": len(grouped[label]),
                        }
                    )

            st.table(table_data)
        else:
            st.info("No entities detected. Try a different resume text.")
