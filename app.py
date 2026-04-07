from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGResponse, SourceSnippet, build_custom_rag, build_rag
from risk_model import (
    FEATURE_COLUMNS,
    get_dataset_overview,
    get_dataset_overview_from_bundle,
    predict_risk,
    predict_risk_with_bundle,
    train_risk_model,
    train_risk_model_from_dataframe,
)

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


load_dotenv()

# Streamlit Cloud stores secrets in st.secrets, not .env.
# Mirror key settings into environment variables so the existing pipeline can use os.getenv.
try:
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
    if "OPENAI_CHAT_MODEL" in st.secrets and not os.getenv("OPENAI_CHAT_MODEL"):
        os.environ["OPENAI_CHAT_MODEL"] = str(st.secrets["OPENAI_CHAT_MODEL"])
    if "OPENAI_WEB_MODEL" in st.secrets and not os.getenv("OPENAI_WEB_MODEL"):
        os.environ["OPENAI_WEB_MODEL"] = str(st.secrets["OPENAI_WEB_MODEL"])
except Exception:
    # If secrets are unavailable locally, continue with .env / process environment values.
    pass

st.set_page_config(
    page_title="Hybrid Maternal Health Agent",
    page_icon="🤰",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        background:
            radial-gradient(circle at 12% 80%, rgba(59, 130, 246, 0.32), transparent 18%),
            radial-gradient(circle at 88% 82%, rgba(37, 99, 235, 0.28), transparent 16%),
            radial-gradient(circle at 50% 0%, rgba(29, 78, 216, 0.10), transparent 30%),
            linear-gradient(180deg, #070b22 0%, #08112e 52%, #0a1438 100%);
        color: #e5eefc;
    }
    .block-container {
        padding-top: 1.2rem;
    }
    h1, h2, h3 {
        font-family: Georgia, "Times New Roman", serif;
        color: #f8fbff;
        letter-spacing: -0.02em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(7, 12, 36, 0.55);
        border: 1px solid rgba(98, 125, 255, 0.14);
        padding: 0.35rem;
        border-radius: 999px;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(14, 23, 58, 0.86);
        border: 1px solid rgba(96, 124, 255, 0.10);
        border-radius: 999px;
        color: #a9c2ff;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2457ff, #4f7cff);
        color: white;
        border-color: transparent;
        box-shadow: 0 10px 25px rgba(36, 87, 255, 0.28);
    }
    .hero-shell {
        border: 1px solid rgba(111, 141, 255, 0.16);
        background:
            linear-gradient(120deg, rgba(10, 18, 48, 0.94), rgba(13, 23, 62, 0.92)),
            linear-gradient(135deg, rgba(19, 33, 88, 0.72), rgba(8, 14, 38, 0.94));
        border-radius: 26px;
        padding: 2.2rem 2.4rem 2.4rem 2.4rem;
        box-shadow: 0 24px 70px rgba(3, 8, 25, 0.38);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .brand-mark {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, rgba(36, 87, 255, 0.95), rgba(103, 131, 255, 0.72));
        box-shadow: 0 14px 28px rgba(36, 87, 255, 0.24);
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
    }
    .brand-title {
        color: #f8fbff;
        font-size: 1.25rem;
        font-weight: 800;
    }
    .brand-subtitle {
        color: #8ea9e8;
        font-size: 0.9rem;
    }
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -8% -35% auto;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(69, 121, 255, 0.28), transparent 64%);
        pointer-events: none;
    }
    .hero-kicker {
        color: #8bb4ff;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.8rem;
        line-height: 1.22;
        font-weight: 700;
        max-width: 1120px;
        margin: 0 0 1.25rem 0;
    }
    .hero-copy {
        color: #b7c8ef;
        font-size: 1.15rem;
        line-height: 1.75;
        max-width: 980px;
        margin-bottom: 1.6rem;
    }
    .hero-preview {
        border-radius: 26px;
        border: 1px solid rgba(111, 141, 255, 0.14);
        background: linear-gradient(180deg, rgba(10, 16, 40, 0.98), rgba(13, 22, 54, 0.94));
        padding: 1.45rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 18px 42px rgba(2, 8, 23, 0.34);
        margin-top: 0.8rem;
    }
    .preview-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    .preview-title {
        color: #f8fbff;
        font-weight: 800;
        font-size: 1rem;
    }
    .preview-chip {
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        background: rgba(36, 87, 255, 0.18);
        color: #dce8ff;
        font-size: 0.8rem;
        border: 1px solid rgba(92, 123, 255, 0.2);
    }
    .preview-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1.05rem;
        margin-bottom: 1.25rem;
    }
    .preview-card {
        border-radius: 18px;
        padding: 0.95rem;
        background: rgba(16, 27, 67, 0.94);
        border: 1px solid rgba(111, 141, 255, 0.10);
    }
    .preview-label {
        color: #88a5e6;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .preview-value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.35rem;
    }
    .preview-meta {
        color: #b8c9ef;
        font-size: 0.84rem;
    }
    .preview-bars {
        display: grid;
        gap: 0.65rem;
    }
    .preview-row {
        display: grid;
        grid-template-columns: 100px 1fr 50px;
        align-items: center;
        gap: 0.7rem;
    }
    .preview-row span {
        color: #c7d8ff;
        font-size: 0.86rem;
    }
    .preview-track {
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
    }
    .preview-fill-blue,
    .preview-fill-red,
    .preview-fill-green {
        height: 100%;
        border-radius: 999px;
    }
    .preview-fill-blue { background: linear-gradient(135deg, #2457ff, #63a4ff); width: 83%; }
    .preview-fill-red { background: linear-gradient(135deg, #ef4444, #f97316); width: 61%; }
    .preview-fill-green { background: linear-gradient(135deg, #16a34a, #4ade80); width: 74%; }
    .soft-card {
        border: 1px solid rgba(113, 145, 255, 0.14);
        background: linear-gradient(180deg, rgba(13, 21, 52, 0.92), rgba(10, 17, 44, 0.92));
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 18px 44px rgba(2, 8, 23, 0.32);
    }
    .mode-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.72rem;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
        color: #dce8ff;
        background: rgba(34, 83, 255, 0.22);
        border: 1px solid rgba(93, 125, 255, 0.26);
    }
    .risk-banner {
        border-radius: 22px;
        padding: 1.1rem 1.25rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
    }
    .risk-high { background: linear-gradient(135deg, #b91c1c, #ef4444); }
    .risk-mid { background: linear-gradient(135deg, #1d4ed8, #3b82f6); }
    .risk-low { background: linear-gradient(135deg, #166534, #22c55e); }
    .chart-shell {
        border: 1px solid rgba(112, 139, 255, 0.14);
        border-radius: 24px;
        padding: 1rem 1rem 0.4rem 1rem;
        background: linear-gradient(180deg, rgba(9, 16, 40, 0.94), rgba(10, 18, 46, 0.96));
        box-shadow: 0 22px 52px rgba(2, 8, 23, 0.32);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(12, 20, 50, 0.96), rgba(10, 16, 42, 0.98));
        border: 1px solid rgba(105, 138, 255, 0.14);
        border-radius: 20px;
        padding: 0.9rem 1rem;
        box-shadow: 0 18px 34px rgba(2, 8, 23, 0.26);
    }
    div[data-testid="stMetricLabel"] {
        color: #9ab5ef;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fbff;
        font-weight: 800;
    }
    .stButton > button, .stDownloadButton > button, .stForm button[type="submit"] {
        background: linear-gradient(135deg, #2457ff, #4f7cff);
        color: white;
        border: none;
        border-radius: 14px;
        font-weight: 800;
        box-shadow: 0 14px 28px rgba(36, 87, 255, 0.28);
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stForm button[type="submit"]:hover {
        background: linear-gradient(135deg, #1f46cf, #2457ff);
        color: white;
    }
    .stTextInput input, .stTextArea textarea, [data-baseweb="input"] input {
        border-radius: 16px;
        background: rgba(8, 15, 38, 0.9) !important;
        color: #f8fbff !important;
        border-color: rgba(103, 131, 255, 0.22) !important;
    }
    .stAlert {
        border-radius: 18px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(112, 139, 255, 0.14);
        border-radius: 20px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_agent():
    return build_rag()


@st.cache_resource
def get_risk_bundle():
    return train_risk_model()


def transcribe_audio_file(audio_file) -> str:
    if sr is None:
        raise RuntimeError("speech_recognition is not installed.")

    suffix = Path(audio_file.name).suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_path = Path(temp_file.name)

        recognizer = sr.Recognizer()
        with sr.AudioFile(str(temp_path)) as source:
            audio_data = recognizer.record(source)

        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError as exc:
        raise RuntimeError("I couldn't understand the audio clearly.") from exc
    except sr.RequestError as exc:
        raise RuntimeError(f"Speech recognition service is unavailable: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def synthesize_speech(text: str) -> bytes | None:
    if gTTS is None or not text.strip():
        return None

    audio_buffer = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(audio_buffer)
    return audio_buffer.getvalue()


def render_sources(sources: list[SourceSnippet]) -> None:
    if not sources:
        return

    st.subheader("Sources")
    for index, source in enumerate(sources, start=1):
        title = f"Source {index} - {source.title}"
        if source.location:
            title = f"{title} ({source.location})"
        with st.expander(title):
            st.write(source.content)
            if source.url:
                st.markdown(f"[Open source]({source.url})")


def answer_mode_label(result: RAGResponse) -> str:
    mapping = {
        "pdf": "WHO PDF",
        "dataset": "CSV dataset",
        "web": "Live web search",
        "greeting": "Welcome",
        "empty": "No input",
    }
    return mapping.get(result.answer_mode, result.answer_mode.title())


def risk_banner_class(label: str) -> str:
    if label == "high risk":
        return "risk-high"
    if label == "mid risk":
        return "risk-mid"
    return "risk-low"


def chart_frame(title: str, subtitle: str = "") -> None:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    st.markdown(f"#### {title}")
    if subtitle:
        st.caption(subtitle)


def close_chart_frame() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.properties(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="rgba(148, 163, 184, 0.18)",
            domainColor="rgba(148, 163, 184, 0.28)",
            tickColor="rgba(148, 163, 184, 0.28)",
            labelColor="#c8d8ff",
            titleColor="#f8fbff",
            labelFont="Manrope",
            titleFont="Manrope",
        )
        .configure_title(
            color="#f8fbff",
            font="Source Serif 4",
            anchor="start",
            fontSize=18,
        )
        .configure_legend(
            labelColor="#dbe7ff",
            titleColor="#f8fbff",
            labelFont="Manrope",
            titleFont="Manrope",
            orient="bottom",
        )
    )


UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_custom_bundle(csv_path: str):
    dataset = pd.read_csv(csv_path)
    return train_risk_model_from_dataframe(dataset)


@st.cache_resource
def get_custom_agent(pdf_path: str, csv_path: str | None = None):
    bundle = get_custom_bundle(csv_path) if csv_path else get_risk_bundle()
    return build_custom_rag(pdf_path, bundle)


st.session_state.setdefault("query_text", "")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("custom_pdf_path", "")
st.session_state.setdefault("custom_csv_path", "")

custom_pdf_path = st.session_state["custom_pdf_path"]
custom_csv_path = st.session_state["custom_csv_path"]
use_custom_pdf = bool(custom_pdf_path)
use_custom_csv = bool(custom_csv_path)

bundle = None
agent = None
agent_error = None
try:
    bundle = get_custom_bundle(custom_csv_path) if use_custom_csv else get_risk_bundle()
    if use_custom_pdf:
        agent = get_custom_agent(custom_pdf_path, custom_csv_path if use_custom_csv else None)
    elif use_custom_csv:
        agent = build_custom_rag(Path("data/maternal_health.pdf"), bundle)
    else:
        agent = get_agent()
except Exception as exc:
    agent_error = str(exc)
    if bundle is None:
        bundle = get_risk_bundle()

dataset = bundle.dataset
guidelines_df = agent.get_guidelines_dataframe() if agent is not None else pd.DataFrame(columns=["Page", "Guideline Excerpt"])
dataset_overview = get_dataset_overview_from_bundle(bundle)

tab_home, tab_chat, tab_risk, tab_guidelines = st.tabs(
    ["Home", "Health Assistant (Chat)", "Risk Assessment (AI)", "Medical Guidelines (Tables)"]
)

with tab_home:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .upload-status-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }
        .upload-status-card {
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: rgba(12, 20, 50, 0.9);
            border: 1px solid rgba(105, 138, 255, 0.16);
        }
        .upload-status-label {
            color: #8ea9e8;
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .upload-status-value {
            color: #f8fbff;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .upload-status-meta {
            color: #b7c8ef;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="brand-wrap">
                <div class="brand-mark">MH</div>
                <div>
                    <div class="brand-title">Maternal Health Agent</div>
                    <div class="brand-subtitle">Clinical guidance, risk intelligence, and beautiful analytics</div>
                </div>
            </div>
            <div class="hero-kicker">Hybrid Maternal Health Agent</div>
            <div class="hero-title">Clinical guidance, risk intelligence, and elegant maternal-health analytics.</div>
            <div class="hero-copy">
                Ask questions against the WHO maternal-health PDF, explore the maternal risk dataset, run a bedside-style risk estimate,
                and fall back to trusted web results when the local guideline context is too thin.
            </div>
            <div class="hero-preview">
                <div class="preview-top">
                    <div class="preview-title">Analytics Preview</div>
                    <div class="preview-chip">Model-ready insights</div>
                </div>
                <div class="preview-grid">
                    <div class="preview-card">
                        <div class="preview-label">WHO Guideline Chunks</div>
                        <div class="preview-value">{len(guidelines_df):,}</div>
                        <div class="preview-meta">Searchable excerpts ready for retrieval</div>
                    </div>
                    <div class="preview-card">
                        <div class="preview-label">Dataset Rows</div>
                        <div class="preview-value">{len(dataset):,}</div>
                        <div class="preview-meta">Structured maternal risk cases loaded</div>
                    </div>
                    <div class="preview-card">
                        <div class="preview-label">Model Accuracy</div>
                        <div class="preview-value">{bundle.accuracy:.1%}</div>
                        <div class="preview-meta">Validation performance on risk prediction</div>
                    </div>
                </div>
                <div class="preview-bars">
                    <div class="preview-row">
                        <span>Assistant Quality</span>
                        <div class="preview-track"><div class="preview-fill-blue"></div></div>
                        <span>83%</span>
                    </div>
                    <div class="preview-row">
                        <span>Urgency Detection</span>
                        <div class="preview-track"><div class="preview-fill-red"></div></div>
                        <span>61%</span>
                    </div>
                    <div class="preview-row">
                        <span>Guideline Coverage</span>
                        <div class="preview-track"><div class="preview-fill-green"></div></div>
                        <span>74%</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("WHO Guideline Chunks", f"{len(guidelines_df):,}")
    metric_b.metric("Dataset Rows", f"{len(dataset):,}")
    metric_c.metric("Risk Model Accuracy", f"{bundle.accuracy:.1%}")
    metric_d.metric("Risk Labels", dataset["RiskLevel"].nunique())

    st.markdown("### Welcome")
    st.caption("Clinical guidance, risk intelligence, and elegant maternal-health analytics.")

    home_a, home_b = st.columns(2, gap="large")
    with home_a:
        chart_frame("Risk Distribution", "A quick overview of how the maternal risk labels are distributed.")
        risk_counts = dataset["RiskLevel"].value_counts().rename_axis("RiskLevel").reset_index(name="Count")
        risk_distribution_chart = style_chart(
            alt.Chart(risk_counts)
            .mark_arc(innerRadius=72, outerRadius=116)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "RiskLevel:N",
                    scale=alt.Scale(
                        domain=["low risk", "mid risk", "high risk"],
                        range=["#22c55e", "#3b82f6", "#ef4444"],
                    ),
                    legend=alt.Legend(title="Risk Level"),
                ),
                tooltip=["RiskLevel:N", "Count:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(risk_distribution_chart, width="stretch")
        close_chart_frame()

    with home_b:
        chart_frame("Average Vitals by Risk Level", "Compare the typical profile for low-, mid-, and high-risk groups.")
        overview_long = dataset_overview.melt(
            id_vars="RiskLevel",
            value_vars=FEATURE_COLUMNS,
            var_name="Feature",
            value_name="AverageValue",
        )
        grouped_bar_chart = style_chart(
            alt.Chart(overview_long)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("Feature:N", title="Clinical feature"),
                y=alt.Y("AverageValue:Q", title="Average value"),
                color=alt.Color(
                    "RiskLevel:N",
                    scale=alt.Scale(
                        domain=["low risk", "mid risk", "high risk"],
                        range=["#22c55e", "#3b82f6", "#ef4444"],
                    ),
                    legend=alt.Legend(title="Risk Level"),
                ),
                xOffset="RiskLevel:N",
                tooltip=["RiskLevel:N", "Feature:N", "AverageValue:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(grouped_bar_chart, width="stretch")
        close_chart_frame()

    home_action_a, home_action_b, home_action_c = st.columns(3, gap="large")
    with home_action_a:
        st.markdown("#### Health Assistant")
        st.write("Ask questions against the WHO PDF, the CSV dataset, and trusted live sources when needed.")
    with home_action_b:
        st.markdown("#### Risk Assessment")
        st.write("Enter vitals, get a maternal risk prediction, and retrieve focused guidance for high-risk cases.")
    with home_action_c:
        st.markdown("#### Medical Guidelines")
        st.write("Review extracted WHO excerpts alongside a smaller set of clean, readable analytics.")

    with st.expander("Use Your Own PDF and CSV"):
        st.caption("Optional workspace. The default WHO PDF and default maternal-risk CSV remain active until you explicitly switch one of them.")
        upload_pdf = st.file_uploader("Upload a maternal-health PDF", type=["pdf"], key="custom_pdf_upload")
        upload_csv = st.file_uploader("Upload a maternal-risk CSV", type=["csv"], key="custom_csv_upload")

        st.markdown(
            f"""
            <div class="upload-status-grid">
                <div class="upload-status-card">
                    <div class="upload-status-label">Guideline Source</div>
                    <div class="upload-status-value">{"Custom PDF active" if use_custom_pdf else "Default WHO PDF active"}</div>
                    <div class="upload-status-meta">{"Chat and guideline tables are using your uploaded PDF." if use_custom_pdf else "Chat and guideline tables are still using the built-in WHO PDF."}</div>
                </div>
                <div class="upload-status-card">
                    <div class="upload-status-label">Analytics Source</div>
                    <div class="upload-status-value">{"Custom CSV active" if use_custom_csv else "Default risk CSV active"}</div>
                    <div class="upload-status-meta">{"Risk prediction and charts are using your uploaded CSV." if use_custom_csv else "Risk prediction and charts are still using the built-in maternal-risk CSV."}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action_a, action_b = st.columns(2)
        with action_a:
            if st.button("Activate Uploaded PDF / CSV", use_container_width=True):
                activated_items: list[str] = []

                if upload_pdf is not None:
                    pdf_path = UPLOADS_DIR / "custom_guidelines.pdf"
                    pdf_path.write_bytes(upload_pdf.getvalue())
                    st.session_state["custom_pdf_path"] = str(pdf_path)
                    activated_items.append("PDF")

                if upload_csv is not None:
                    csv_path = UPLOADS_DIR / "custom_dataset.csv"
                    csv_path.write_bytes(upload_csv.getvalue())
                    st.session_state["custom_csv_path"] = str(csv_path)
                    activated_items.append("CSV")

                if activated_items:
                    st.success(
                        f"Activated uploaded {' and '.join(activated_items)} for this session. The default system remains in place for anything you did not replace."
                    )
                    st.rerun()
                else:
                    st.warning("Upload a PDF, a CSV, or both before activating custom files.")

        with action_b:
            if st.button("Reset to Default Data", use_container_width=True):
                st.session_state["custom_pdf_path"] = ""
                st.session_state["custom_csv_path"] = ""
                st.success("Default WHO PDF and default CSV restored.")
                st.rerun()

        if use_custom_pdf or use_custom_csv:
            active_modes = []
            if use_custom_pdf:
                active_modes.append("custom PDF")
            if use_custom_csv:
                active_modes.append("custom CSV")
            st.info(f"Custom data mode is active for this session: {', '.join(active_modes)}.")

with tab_chat:
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.markdown("### Ask the agent")
        st.caption("The assistant routes between the WHO PDF, the risk CSV, and live web search when local context is weak.")

        with st.form("chat_form", clear_on_submit=False):
            question = st.text_area(
                "Question",
                key="query_text",
                height=120,
                placeholder="Ask about maternal warning signs, blood pressure, dataset patterns, or the latest guidance.",
            )
            submitted = st.form_submit_button("Ask the Assistant")

        if submitted and question.strip():
            if agent is None:
                st.error(f"Unable to initialize the assistant right now: {agent_error}")
            else:
                try:
                    with st.spinner("Reviewing guidelines, data, and fallback sources..."):
                        result = agent.ask(question)
                    st.session_state["chat_history"].insert(0, {"question": question, "result": result})
                except Exception as exc:
                    st.error(f"Unable to answer your question right now: {exc}")

        history = st.session_state["chat_history"]
        if history:
            latest = history[0]
            result = latest["result"]
            st.markdown(f'<div class="mode-pill">{answer_mode_label(result)}</div>', unsafe_allow_html=True)
            st.markdown("### Answer")
            st.write(result.answer)

            audio_bytes = synthesize_speech(result.answer)
            if audio_bytes:
                st.subheader("Listen")
                st.audio(audio_bytes, format="audio/mp3")

            render_sources(result.sources)

            if len(history) > 1:
                st.markdown("### Recent questions")
                for item in history[1:4]:
                    with st.expander(item["question"]):
                        st.write(item["result"].answer)

    with right:
        st.markdown("### Voice input")
        if sr is None:
            st.info("Install `SpeechRecognition` to enable transcription.")
        else:
            audio_file = None
            if hasattr(st, "audio_input"):
                audio_file = st.audio_input("Record a question")
            if audio_file is None:
                audio_file = st.file_uploader(
                    "Upload a voice recording",
                    type=["wav", "aiff", "aif", "flac"],
                    key="chat_audio_upload",
                )

            if audio_file is not None and st.button("Use Voice Question"):
                try:
                    with st.spinner("Transcribing your question..."):
                        st.session_state["query_text"] = transcribe_audio_file(audio_file)
                    st.success("Voice input added to the question box.")
                except Exception as exc:
                    st.error(f"Unable to transcribe audio right now: {exc}")

        st.markdown("### Suggested prompts")
        st.markdown(
            """
            - What are urgent maternal danger signs during pregnancy?
            - What patterns stand out in the maternal risk dataset?
            - Latest WHO advice on postpartum warning signs
            """
        )

        if agent is None:
            st.warning(f"Assistant initialization issue: {agent_error}")

with tab_risk:
    st.markdown("### AI-assisted maternal risk assessment")
    st.caption("This is a decision-support tool, not a diagnosis. High-risk results will trigger guideline retrieval automatically.")

    medians = dataset[FEATURE_COLUMNS].median().to_dict()
    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=60, value=int(medians["Age"]))
            systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=200, value=int(medians["SystolicBP"]))
        with col2:
            diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=140, value=int(medians["DiastolicBP"]))
            blood_sugar = st.number_input("Blood Sugar (BS)", min_value=4.0, max_value=25.0, value=float(medians["BS"]), step=0.1)
        with col3:
            body_temp = st.number_input("Body Temperature", min_value=95.0, max_value=105.0, value=float(medians["BodyTemp"]), step=0.1)
            heart_rate = st.number_input("Heart Rate", min_value=40, max_value=150, value=int(medians["HeartRate"]))

        risk_submitted = st.form_submit_button("Run Risk Assessment")

    if risk_submitted:
        patient_features = {
            "Age": float(age),
            "SystolicBP": float(systolic_bp),
            "DiastolicBP": float(diastolic_bp),
            "BS": float(blood_sugar),
            "BodyTemp": float(body_temp),
            "HeartRate": float(heart_rate),
        }
        prediction = predict_risk_with_bundle(patient_features, bundle)

        st.markdown(
            f"""
            <div class="risk-banner {risk_banner_class(prediction.label)}">
                <div style="font-size:0.82rem; letter-spacing:0.08em; text-transform:uppercase; opacity:0.88;">Predicted maternal risk</div>
                <div style="font-size:2rem; font-weight:700; margin-top:0.2rem;">{prediction.label.title()}</div>
                <div style="margin-top:0.35rem; font-size:1rem;">Confidence: {prediction.confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_df = pd.DataFrame(
            {
                "RiskLevel": list(prediction.probabilities.keys()),
                "Probability": list(prediction.probabilities.values()),
            }
        )
        probability_chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
            .encode(
                x=alt.X("RiskLevel:N", sort="-y", title="Risk level"),
                y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), title="Model probability"),
                color=alt.Color(
                    "RiskLevel:N",
                    scale=alt.Scale(
                        domain=["low risk", "mid risk", "high risk"],
                        range=["#16a34a", "#2563eb", "#dc2626"],
                    ),
                    legend=None,
                ),
            )
            .properties(height=260)
        )
        st.altair_chart(probability_chart, width="stretch")

        if prediction.label == "high risk":
            guidance_query = (
                "Based on WHO maternal health guidance, what urgent monitoring, referral advice, and warning signs "
                f"are relevant for a pregnant patient with age {age}, systolic BP {systolic_bp}, diastolic BP {diastolic_bp}, "
                f"blood sugar {blood_sugar}, body temperature {body_temp}, and heart rate {heart_rate}?"
            )
            if agent is None:
                st.error(f"Risk prediction worked, but the assistant could not initialize: {agent_error}")
            else:
                try:
                    with st.spinner("Fetching WHO guidance for this high-risk profile..."):
                        guidance_result = agent.ask(guidance_query)
                    st.markdown("### Recommended guidance")
                    st.write(guidance_result.answer)
                    render_sources(guidance_result.sources)
                except Exception as exc:
                    st.error(f"Risk was predicted successfully, but guideline retrieval failed: {exc}")
        else:
            st.info("This profile was not classified as high risk. Continue monitoring and review the guideline table for reference.")

with tab_guidelines:
    st.markdown("### Medical Guidelines")
    st.caption("Focused reference tables and a smaller set of visual summaries.")

    search_term = st.text_input("Filter guideline excerpts", placeholder="Try: fever, bleeding, hypertension, postpartum")
    filtered_guidelines = guidelines_df
    if search_term.strip():
        filtered_guidelines = guidelines_df[
            guidelines_df["Guideline Excerpt"].str.contains(search_term, case=False, na=False)
        ]

    st.dataframe(filtered_guidelines, width="stretch", hide_index=True)

    analytics_a, analytics_b = st.columns(2, gap="large")

    with analytics_a:
        chart_frame("Risk Distribution", "A quick view of how the maternal risk labels are distributed in the dataset.")
        risk_counts = (
            dataset["RiskLevel"]
            .value_counts()
            .rename_axis("RiskLevel")
            .reset_index(name="Count")
        )
        risk_distribution_chart = style_chart(
            alt.Chart(risk_counts)
            .mark_arc(innerRadius=72, outerRadius=116)
            .encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color(
                    "RiskLevel:N",
                    scale=alt.Scale(
                        domain=["low risk", "mid risk", "high risk"],
                        range=["#22c55e", "#3b82f6", "#ef4444"],
                    ),
                    legend=alt.Legend(title="Risk Level"),
                ),
                tooltip=["RiskLevel:N", "Count:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(risk_distribution_chart, width="stretch")
        close_chart_frame()

    with analytics_b:
        chart_frame("Blood Pressure vs Blood Sugar", "This scatter view helps users spot how higher-risk cases cluster across vital signals.")
        scatter_chart = style_chart(
            alt.Chart(dataset)
            .mark_circle(size=95, opacity=0.78)
            .encode(
                x=alt.X("SystolicBP:Q", title="Systolic BP"),
                y=alt.Y("BS:Q", title="Blood Sugar (BS)"),
                color=alt.Color(
                    "RiskLevel:N",
                    scale=alt.Scale(
                        domain=["low risk", "mid risk", "high risk"],
                        range=["#22c55e", "#3b82f6", "#ef4444"],
                    ),
                    legend=alt.Legend(title="Risk Level"),
                ),
                size=alt.Size("HeartRate:Q", title="Heart Rate"),
                tooltip=["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"],
            )
            .properties(height=320)
        )
        st.altair_chart(scatter_chart, width="stretch")
        close_chart_frame()

    with st.expander("Dataset Quick Reference"):
        overview_left, overview_right = st.columns([1.1, 0.9], gap="large")
        with overview_left:
            st.dataframe(dataset_overview, width="stretch", hide_index=True)
        with overview_right:
            st.dataframe(dataset.head(20), width="stretch", hide_index=True)
