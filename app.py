"""
Streamlit Web Application for Research Paper Quality Evaluation.
Uses pre-trained LightGBM regressor models and SentenceTransformer text embeddings.
"""

import datetime
import json
import logging
from pathlib import Path
import streamlit as st

from src.features import clean_abstract, prepare_features, load_categories
from src.utils import extract_text_from_pdf, find_similar_papers
from src.predictor import ResearchPaperPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# PAGE CONFIG (Wide layout to allow more horizontal space)
st.set_page_config(page_title="Research Paper Quality Analyzer", layout="wide")

# ARXIV CATEGORIES FRIENDLY NAME MAPPING
CATEGORY_NAMES = {
    "cs.LG": "Machine Learning (cs.LG)",
    "cs.CV": "Computer Vision (cs.CV)",
    "cs.CL": "NLP / Language (cs.CL)",
    "cs.AI": "Artificial Intelligence (cs.AI)",
    "stat.ML": "Statistical Machine Learning (stat.ML)",
    "cs.NE": "Neural & Evolutionary Computing (cs.NE)",
    "cs.CR": "Cryptography & Security (cs.CR)",
    "cs.CY": "Computers & Society (cs.CY)",
    "cs.DC": "Distributed/Parallel Computing (cs.DC)",
    "cs.DL": "Digital Libraries (cs.DL)",
    "cs.HC": "Human-Computer Interaction (cs.HC)",
    "cs.IT": "Information Theory (cs.IT)",
    "cs.SD": "Sound/Audio Processing (cs.SD)",
    "cs.SE": "Software Engineering (cs.SE)",
    "eess.SY": "Systems & Control (eess.SY)",
    "math.CO": "Combinatorics (math.CO)",
    "math.OC": "Optimization & Control (math.OC)",
    "quant-ph": "Quantum Physics (quant-ph)",
    "astro-ph.GA": "Astrophysics of Galaxies (astro-ph.GA)",
    "cond-mat.supr-con": "Superconductivity (cond-mat.supr-con)",
    "unknown": "Other / Unknown"
}

# PREMIUM STYLE (Wide, Professional Typography, Centered Contents)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
}
.title-container {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.title-text {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle-text {
    color: #94a3b8;
    font-size: 19px;
    margin-top: 0.75rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}
h3 {
    font-size: 30px !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    margin-top: 3rem !important;
    margin-bottom: 1.25rem !important;
}
.stMarkdown p {
    font-size: 16px !important;
    line-height: 1.6 !important;
}
.metric-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #334155;
    padding: 2rem 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    backdrop-filter: blur(10px);
    text-align: center;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.metric-title {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.75rem;
}
.subscore-value {
    font-size: 28px;
    font-weight: 800;
    color: #818cf8;
}
.tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 8px;
    background: #334155;
    color: #e2e8f0;
    margin-right: 6px;
    font-size: 12px;
    font-weight: 500;
}
.similar-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid #334155;
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="title-container">
    <div class="title-text">Research Paper Quality Analyzer</div>
    <div class="subtitle-text">An AI/ML-powered system for evaluating research papers and discovering similar research.</div>
</div>
""", unsafe_allow_html=True)

# CACHED LOADERS
@st.cache_resource
def load_ml_predictor() -> ResearchPaperPredictor:
    """
    Caches and loads the core prediction wrapper containing the LightGBM models.
    """
    return ResearchPaperPredictor(models_dir="models")


@st.cache_resource
def load_sentence_transformer():
    """
    Caches and loads the SentenceTransformer embedder.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer: {e}")
        st.error(f"Could not load SentenceTransformer model: {e}")
        return None


# LOAD INSTANCES
predictor = load_ml_predictor()
embedder = load_sentence_transformer()

# DYNAMIC DATES & CATEGORIES
db_dataset_path = "data/embeddingdataset.json"
categories = load_categories(db_dataset_path)
ANNOTATIONS_PATH = Path("data/annotations.json")


def load_annotation_records() -> dict:
    """Load saved human annotations without changing the source dataset."""
    if not ANNOTATIONS_PATH.exists():
        return {}
    try:
        with ANNOTATIONS_PATH.open("r", encoding="utf-8") as file:
            annotations = json.load(file)
        if not isinstance(annotations, dict):
            raise ValueError("Annotation file must contain a JSON object.")
        return annotations
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not load annotations: {error}")
        return {}


def save_annotation(arxiv_id: str, scores: dict) -> None:
    """Persist one annotation keyed by the paper's stable ArXiv ID."""
    annotations = load_annotation_records()
    annotations[arxiv_id] = scores
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ANNOTATIONS_PATH.open("w", encoding="utf-8") as file:
        json.dump(annotations, file, indent=2)


def render_annotation_screen(records: list) -> None:
    """Render a manual 1–5 annotation workflow for currently unlabeled papers."""
    annotations = load_annotation_records()
    unlabeled_records = [
        record for record in records
        if all(record.get(field, 0) == 0 for field in (
            "manual_score", "novelty_score", "rigor_score", "impact_score"
        ))
        and record.get("arxiv_id") not in annotations
    ]

    st.header("Human Annotation")
    st.caption(
        "Score one paper at a time from 1 (lowest) to 5 (highest). "
        "Annotations are saved separately in data/annotations.json."
    )
    if not unlabeled_records:
        st.success("No unannotated placeholder records remain.")
        return

    paper_options = {
        f"{record.get('title', 'Untitled')} ({record.get('arxiv_id', 'unknown')})": record
        for record in unlabeled_records
    }
    selected_label = st.selectbox("Paper to annotate", list(paper_options))
    paper = paper_options[selected_label]

    st.subheader(paper.get("title", "Untitled"))
    st.caption(
        f"Category: {paper.get('primary_category', 'unknown')} | "
        f"Authors: {paper.get('author_count', 'unknown')} | "
        f"Abstract length: {paper.get('summary_len', 0)} characters"
    )
    with st.expander("Read abstract", expanded=True):
        st.write(paper.get("summary", "No abstract available."))

    overall = st.slider("Overall quality", 1, 5, 3, key=f"overall_{paper['arxiv_id']}")
    novelty = st.slider("Novelty", 1, 5, 3, key=f"novelty_{paper['arxiv_id']}")
    rigor = st.slider("Rigor", 1, 5, 3, key=f"rigor_{paper['arxiv_id']}")
    impact = st.slider("Impact", 1, 5, 3, key=f"impact_{paper['arxiv_id']}")
    notes = st.text_area("Optional annotation notes", key=f"notes_{paper['arxiv_id']}")

    if st.button("Save annotation", type="primary"):
        save_annotation(
            paper["arxiv_id"],
            {
                "arxiv_id": paper["arxiv_id"],
                "title": paper.get("title", "Untitled"),
                "overall": overall,
                "novelty": novelty,
                "rigor": rigor,
                "impact": impact,
                "notes": notes,
            },
        )
        st.success("Annotation saved. Select another paper to continue.")
        st.rerun()

# MAIN CONTAINER WITH DYNAMIC WIDTH
col_left, col_center, col_right = st.columns([1, 8, 1])

with col_center:
    with st.sidebar:
        app_mode = st.radio(
            "Application mode",
            ["Analyze paper", "Annotate dataset"],
            help="Use Annotate dataset to add verified 1–5 labels to placeholder records.",
        )

    if app_mode == "Annotate dataset":
        try:
            with open(db_dataset_path, "r", encoding="utf-8") as f:
                render_annotation_screen(json.load(f))
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not load annotation dataset: {error}")
        st.stop()

    # SECTION 1: UPLOAD AREA (Directly below subtitle, no empty containers)
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf", label_visibility="visible")

    if uploaded_file:
        # Success indicator & File details
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.success(f"📄 Successfully loaded: **{uploaded_file.name}** ({file_size_kb:.1f} KB)")
        
        if not predictor.is_ready() or embedder is None:
            st.warning("Prediction models or sentence embedders are currently unavailable. Running fallback heuristic.")
            
            with st.spinner("Analyzing text layout..."):
                text_content = extract_text_from_pdf(uploaded_file)
                heur_score = min(50 + (len(text_content) // 1200) + ("abstract" in text_content.lower()) * 10, 100)
                
                # Center heuristic display
                st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 3px solid #38bdf8; padding: 2.5rem; border-radius: 24px; display: inline-block; min-width: 340px; text-align: center; box-shadow: 0 10px 25px -3px rgba(56, 189, 248, 0.15); backdrop-filter: blur(10px);">
                    <div style="color: #94a3b8; font-size: 18px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;">Overall Quality Score (Heuristic)</div>
                    <div style="font-size: 76px; font-weight: 800; color: #38bdf8; margin-top: 0.5rem; letter-spacing: -0.03em; line-height: 1.1;">{heur_score} <span style="font-size: 32px; color: #94a3b8;">/ 100</span></div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing paper semantics and evaluating metrics..."):
                # 1. Parse text from PDF stream
                full_text = extract_text_from_pdf(uploaded_file)
                abstract = clean_abstract(full_text)
                
                # 2. Vectorize abstract semantic structure (outputs 384 floats)
                abstract_embedding = embedder.encode(abstract).tolist()
                
                # 3. Format structured metadata using neutral model-safe defaults
                # We use years_since_published = 9.976 (dataset median) to prevent score suppression.
                metadata_features = {
                    "author_count": 3.0,                  # Dataset average author count
                    "summary_len": len(abstract),
                    "text_len": len(full_text),
                    "years_since_published": 9.976,       # Neutral dataset-derived median age
                    "primary_category": "cs.LG"           # Most common category
                }
                
                # 4. Format features row
                X_features = prepare_features(metadata_features, abstract_embedding, categories)
                
                # 5. Model score predictions
                predictions = predictor.predict_scores(X_features)

            # SECTION 2: QUALITY SCORE (Hero Centered Display, Extremely Visual and Dominant)
            st.markdown("<div style='text-align: center; margin: 3rem 0 3.5rem 0;'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.85); border: 3px solid #38bdf8; padding: 3rem; border-radius: 24px; display: inline-block; min-width: 360px; text-align: center; box-shadow: 0 15px 35px -5px rgba(56, 189, 248, 0.25); backdrop-filter: blur(10px);">
                <div style="color: #94a3b8; font-size: 18px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">Overall Quality Score</div>
                <div style="font-size: 76px; font-weight: 800; color: #38bdf8; margin-top: 0.5rem; letter-spacing: -0.03em; line-height: 1.1;">{predictions["overall"]:.1f} <span style="font-size: 32px; color: #94a3b8;">/ 100</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # SECTION 3: SCORE BREAKDOWN (Horizontal row of cards)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" title="Innovation Score">
                    <div class="metric-title">Innovation Score</div>
                    <div class="subscore-value">{predictions["novelty"]:.1f} <span style="font-size:12px;color:#94a3b8;">/ 35</span></div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 0.75rem; line-height: 1.4; font-weight: 400;">Measures originality and uniqueness of the research.</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card" title="Methodology Score">
                    <div class="metric-title">Methodology Score</div>
                    <div class="subscore-value">{predictions["rigor"]:.1f} <span style="font-size:12px;color:#94a3b8;">/ 35</span></div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 0.75rem; line-height: 1.4; font-weight: 400;">Evaluates the strength and validity of research methods.</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card" title="Impact Score">
                    <div class="metric-title">Impact Score</div>
                    <div class="subscore-value">{predictions["impact"]:.1f} <span style="font-size:12px;color:#94a3b8;">/ 35</span></div>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 0.75rem; line-height: 1.4; font-weight: 400;">Estimates the projected influence and utility of the paper.</div>
                </div>
                """, unsafe_allow_html=True)

            # SECTION 4: SCORE EXPLANATION
            st.markdown("<h3 style='margin-bottom: 0.5rem;'>🔍 Why Did The Model Give This Score?</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #94a3b8; font-size: 14px; margin-bottom: 1.5rem; font-weight: 400;'>The chart below shows which factors most influenced the predicted quality score.</p>", unsafe_allow_html=True)
            fig_shap = predictor.generate_shap_plot(X_features)
            st.pyplot(fig_shap)
            
            # SECTION 5: SIMILAR PAPERS (Related Research)
            st.markdown("<h3 style='margin-bottom: 1rem;'>🔗 Related Research Papers</h3>", unsafe_allow_html=True)
            try:
                with open(db_dataset_path, "r", encoding="utf-8") as f:
                    db_records = json.load(f)
                
                recommendations = find_similar_papers(abstract_embedding, db_records, top_n=3)
                rec_cols = st.columns(3)
                
                for idx, (sim_score, paper) in enumerate(recommendations):
                    with rec_cols[idx]:
                        st.markdown(f"""
                        <div class="similar-card">
                            <div style="font-weight: 600; font-size: 14px; color: #e2e8f0; line-height: 1.4; min-height: 56px;">{paper.get('title', 'Untitled')}</div>
                            <div style="font-size: 14px; color: #38bdf8; font-weight: 700; margin-top: 0.75rem;">Similarity: {sim_score*100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as err:
                st.caption(f"Could not load similar papers: {err}")

            # SECTION 6: ABSTRACT (Collapsed by default)
            with st.expander("📝 Extracted Research Abstract", expanded=False):
                st.write(abstract)
                st.caption(f"Full Character Count: {len(full_text)} | Abstract Word Count: {len(abstract.split())}")
    else:
        # Default view: Sample papers list
        st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>📋 Reference Paper Library</h3>", unsafe_allow_html=True)
        
        try:
            with open(db_dataset_path, "r", encoding="utf-8") as f:
                db_records = json.load(f)
            
            # Dynamic Category Filter
            available_cats = sorted(list(set(p.get("primary_category", "unknown") for p in db_records if p.get("primary_category"))))
            selected_filter = st.selectbox(
                "Filter reference library by category", 
                ["All"] + available_cats,
                format_func=lambda x: CATEGORY_NAMES.get(x, x) if x != "All" else "All Reference Papers"
            )
            
            filtered_records = db_records
            if selected_filter != "All":
                filtered_records = [p for p in db_records if p.get("primary_category") == selected_filter]
                
            sorted_records = sorted(filtered_records, key=lambda x: x.get("manual_score", 0), reverse=True)
            
            grid_cols = st.columns(2)
            for idx, paper in enumerate(sorted_records[:6]):
                col_idx = idx % 2
                with grid_cols[col_idx]:
                    friendly_cat = CATEGORY_NAMES.get(paper.get('primary_category', 'unknown'), paper.get('primary_category', 'unknown'))
                    st.markdown(f"""
                    <div class="metric-card" style="min-height: 150px; justify-content: space-between;">
                        <div style="font-weight: 700; font-size: 15px; color: #e2e8f0; text-align: left; line-height: 1.4;">{paper.get('title', 'Untitled')}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                            <span class="tag" style="background: #1e293b;">{friendly_cat}</span>
                            <span style="font-size: 18px; font-weight: 800; color: #38bdf8;">{paper.get('manual_score')} / 100</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as err:
            st.error(f"Error loading dataset: {err}")