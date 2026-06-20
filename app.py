"""Streamlit frontend for Resume RAG — ATS-style resume evaluation."""

from __future__ import annotations

import os
import re

import streamlit as st
from dotenv import load_dotenv

from rag_engine import analyze_all_resumes, extract_text_from_bytes

load_dotenv()

st.set_page_config(
    page_title="Resume RAG Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        color: white !important;
    }

    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        color: #e2e8f0 !important;
    }

    .metric-card {
        background: linear-gradient(145deg, #f8fafc, #f1f5f9);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0ea5e9;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    .result-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .step-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background: #f8fafc;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        border: none;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1);
    }
</style>
"""

DEFAULT_JOB_DESCRIPTION = """We are looking for a Python Developer with experience in:
- Python
- Machine Learning
- TensorFlow
- NLP
- SQL
- REST APIs
- Git
- Cloud deployment
"""


def parse_match_percentage(evaluation: str) -> int | None:
    """Extract match percentage from LLM evaluation text."""
    patterns = [
        r"Match Percentage[:\s*]*\*{0,2}\s*(\d+)\s*%",
        r"(\d+)\s*%\s*(?:match|Match)",
    ]
    for pattern in patterns:
        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), 100)
    return None


def score_color(pct: int) -> str:
    if pct >= 75:
        return "#16a34a"
    if pct >= 50:
        return "#ca8a04"
    return "#dc2626"


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>📄 Resume RAG Analyzer</h1>
            <p>Upload resumes, paste a job description, and get AI-powered ATS evaluations using RAG.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, float]:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        api_key = st.text_input(
            "Google API Key",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
            help="Get your key from Google AI Studio. Stored only for this session.",
        )

        model = st.selectbox(
            "Gemini Model",
            options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
            index=0,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower = more focused and consistent responses.",
        )

        st.markdown("---")
        st.markdown("### How it works")
        st.markdown(
            """
            1. **Upload** PDF resumes
            2. **Paste** the job description
            3. **Analyze** — chunks are embedded & retrieved via FAISS
            4. **Review** ATS-style match reports powered by Gemini
            """
        )

    return api_key, model, temperature


def render_results(results: list) -> None:
    st.markdown("## 📊 Analysis Results")

    cols = st.columns(min(len(results), 4))
    for idx, result in enumerate(results):
        pct = parse_match_percentage(result.evaluation)
        with cols[idx % len(cols)]:
            if pct is not None:
                color = score_color(pct)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {color};">{pct}%</div>
                        <div class="metric-label">{result.resume_name[:28]}{"..." if len(result.resume_name) > 28 else ""}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric(label=result.resume_name[:20], value="Done")

    st.markdown("---")

    for result in results:
        pct = parse_match_percentage(result.evaluation)
        badge = f" — **{pct}% match**" if pct is not None else ""

        with st.expander(f"📋 {result.resume_name}{badge}", expanded=len(results) == 1):
            st.markdown(result.evaluation)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_header()

    api_key, model, temperature = render_sidebar()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<span class="step-badge">Step 1</span> Upload Resumes', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Select one or more PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} resume(s) ready for analysis")
            with st.expander("View uploaded files"):
                for f in uploaded_files:
                    st.write(f"• {f.name} ({f.size / 1024:.1f} KB)")

    with col_right:
        st.markdown('<span class="step-badge">Step 2</span> Job Description', unsafe_allow_html=True)
        job_description = st.text_area(
            "Paste the job description",
            value=DEFAULT_JOB_DESCRIPTION.strip(),
            height=280,
            label_visibility="collapsed",
            placeholder="Paste the full job description here...",
        )

    st.markdown("---")
    st.markdown('<span class="step-badge">Step 3</span> Run Analysis', unsafe_allow_html=True)

    analyze_clicked = st.button("🚀 Analyze Resumes", type="primary", use_container_width=False)

    if analyze_clicked:
        if not uploaded_files:
            st.error("Please upload at least one PDF resume.")
            st.stop()

        if not job_description.strip():
            st.error("Please provide a job description.")
            st.stop()

        if not api_key.strip():
            st.error("Please enter your Google API key in the sidebar.")
            st.stop()

        resume_data: dict[str, str] = {}
        progress = st.progress(0, text="Extracting text from PDFs...")

        for i, uploaded in enumerate(uploaded_files):
            text = extract_text_from_bytes(uploaded.read())
            if not text:
                st.warning(f"Could not extract text from **{uploaded.name}**. Skipping.")
                continue
            resume_data[uploaded.name] = text
            progress.progress((i + 1) / len(uploaded_files) * 0.3, text=f"Extracted {uploaded.name}")

        if not resume_data:
            st.error("No readable text found in the uploaded PDFs.")
            st.stop()

        progress.progress(0.35, text="Building embeddings & vector store...")
        try:
            with st.spinner("Running RAG pipeline — this may take a minute on first run..."):
                progress.progress(0.5, text="Analyzing resumes with Gemini...")
                results = analyze_all_resumes(
                    resume_data=resume_data,
                    job_description=job_description.strip(),
                    model=model,
                    temperature=temperature,
                    api_key=api_key.strip(),
                )
                progress.progress(1.0, text="Complete!")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info("Check your API key, model name, and network connection.")
            st.stop()

        st.session_state["results"] = results
        st.session_state["resume_count"] = len(resume_data)

    if "results" in st.session_state:
        render_results(st.session_state["results"])


if __name__ == "__main__":
    main()
