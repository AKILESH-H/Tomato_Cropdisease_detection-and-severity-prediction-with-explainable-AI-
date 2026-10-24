
from pathlib import Path
from datetime import datetime
import contextlib
import hashlib
import io
import json
import uuid
import html

import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
UPLOADS = RESULTS / "uploads"

RESULTS.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

DISEASE_MODEL_PATH = MODELS / "disease_model_10class.keras"
DISEASE_CLASSES_PATH = MODELS / "disease_classes_10class.json"

st.set_page_config(
    page_title="Plant Disease Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# STYLING
# ---------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700;800&display=swap');

:root {
    --purple: #6241B5;
    --dark: #402779;
    --light: #F4F0FF;
    --text: #28223B;
    --muted: #625A73;
    --border: #E2DAF3;
}

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}

.stApp {
    background: #F8F7FC;
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding: 2rem clamp(1rem, 3vw, 3rem) 3rem;
}

h1, h2, h3 {
    color: var(--dark) !important;
    font-weight: 700 !important;
}

p, li, label {
    line-height: 1.65;
}

[data-testid="stSidebar"] {
    background: #F0EBFF;
    border-right: 1px solid #DDD2F7;
}

[data-testid="stSidebar"] * {
    color: #30234F;
}

.hero {
    background: linear-gradient(120deg, #402779, #7652D2);
    border-radius: 20px;
    padding: clamp(24px, 4vw, 40px);
    margin-bottom: 26px;
    box-shadow: 0 8px 25px rgba(64,39,121,.14);
}

.hero h1 {
    color: white !important;
    font-size: clamp(27px, 3vw, 39px);
    margin: 0 0 10px;
}

.hero p {
    color: #FFFFFF !important;
    font-size: 15px;
    margin: 0;
}

.section-title {
    color: #402779;
    font-size: 23px;
    font-weight: 800;
    margin: 25px 0 14px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-color: #E2DAF3;
    border-radius: 16px;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #A58BD9;
    border-radius: 13px;
    padding: 12px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F7F4FF;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderDropzone"] span {
    color: #30234F !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #6241B5;
    color: white !important;
    border: none;
}

.stButton > button[kind="primary"] {
    background: #6241B5;
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
}

.stButton > button[kind="primary"]:hover {
    background: #402779;
}

.stDownloadButton > button {
    background: white;
    color: #402779 !important;
    border: 1px solid #CDBDEA;
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #402779 !important;
}

[data-testid="stAlert"] p {
    color: #30234F !important;
}

.metric-card {
    background: white;
    border: 1px solid #E2DAF3;
    border-radius: 15px;
    padding: 19px;
    min-height: 125px;
    box-shadow: 0 4px 14px rgba(64,39,121,.05);
    overflow-wrap: anywhere;
}

.metric-label {
    color: #625A73;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .5px;
    margin-bottom: 10px;
}

.metric-value {
    color: #402779;
    font-size: 21px;
    font-weight: 800;
    line-height: 1.4;
}

.metric-help {
    color: #625A73;
    font-size: 12px;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #625A73;
    font-size: 12px;
    padding-top: 28px;
    line-height: 1.8;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem; }
    .hero { padding: 22px; }
    .section-title { font-size: 20px; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD PIPELINE: NEW 10-CLASS MODEL + EXISTING SEVERITY MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_pipeline():
    # Import the existing analysis module. Its severity model,
    # Grad-CAM helper, recommendations and report generation remain.
    import src.analyze_tomato as analyzer

    if not DISEASE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing 10-class model: {DISEASE_MODEL_PATH}"
        )

    if not DISEASE_CLASSES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing class list: {DISEASE_CLASSES_PATH}"
        )

    # Use the new model only for disease classification.
    analyzer.disease_model = tf.keras.models.load_model(
        DISEASE_MODEL_PATH
    )

    with DISEASE_CLASSES_PATH.open("r", encoding="utf-8") as file:
        analyzer.disease_classes = json.load(file)

    if len(analyzer.disease_classes) != 10:
        raise ValueError(
            "Expected 10 disease classes in disease_classes_10class.json."
        )

    return analyzer.analyze


def report_value(report, prefix):
    for line in report.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return "Not available"


def metric_card(label, value, help_text):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-help">{html.escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

def display_results(report, heatmap_path, report_path):
    disease = report_value(report, "Disease prediction")
    disease_conf = report_value(report, "Disease model confidence")
    severity = report_value(report, "Severity prediction")
    severity_conf = report_value(report, "Severity model confidence")
    health = report_value(report, "Prototype health score")

    st.markdown(
        '<div class="section-title">Analysis results</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        metric_card("DETECTED DISEASE", disease, "10-class disease model")

    with c2:
        metric_card("PREDICTED SEVERITY", severity, "Estimated severity category")

    with c3:
        metric_card("PROTOTYPE HEALTH SCORE", health, "Heuristic estimate")

    st.write("")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        metric_card(
            "DISEASE CONFIDENCE",
            disease_conf,
            "Model confidence, not test accuracy",
        )

    with c2:
        metric_card(
            "SEVERITY CONFIDENCE",
            severity_conf,
            "Confidence from the separate severity model",
        )

    st.write("")

    st.warning(
        "Disease classification and severity estimation use separate models. "
        "Their predictions may disagree. The health score is a prototype "
        "heuristic, not a validated measurement of plant health."
    )

    # Visual explanation
    st.markdown(
        '<div class="section-title">Visual explanation</div>',
        unsafe_allow_html=True,
    )

    image_col, explanation_col = st.columns([1.5, 1], gap="large")

    with image_col:
        with st.container(border=True):
            st.subheader("Grad-CAM heatmap")

            if heatmap_path.is_file():
                st.image(
                    str(heatmap_path),
                    use_container_width=True,
                )
            else:
                st.warning("Grad-CAM image was not found.")

            st.caption(
                "Warm colors indicate regions that contribute more strongly "
                "to the model's prediction. They do not prove infection or "
                "biological causation."
            )

    with explanation_col:
        with st.container(border=True):
            st.subheader("Understanding the results")

            st.markdown("**Disease prediction**")
            st.write(
                "The highest-scoring class among the ten supported tomato "
                "disease categories."
            )

            st.markdown("**Severity prediction**")
            st.write(
                "A separate model estimates the affected-area severity class."
            )

            st.markdown("**Grad-CAM**")
            st.write(
                "Highlights image regions influential to the disease model."
            )

            st.markdown("**Health score**")
            st.write(
                "A score mapped from predicted severity; it is not a direct "
                "measurement of plant health."
            )

    # Recommendations
    st.markdown(
        '<div class="section-title">Care and prevention</div>',
        unsafe_allow_html=True,
    )

    lines = report.splitlines()

    try:
        start = lines.index("CARE AND PREVENTION SUGGESTIONS")
        end = lines.index("LIMITATIONS")
        recommendations = lines[start + 1:end]

        with st.container(border=True):
            for line in recommendations:
                item = line.removeprefix("- ").strip()
                if item:
                    st.markdown(f"- {item}")

    except ValueError:
        st.info("Recommendations are available in the technical report.")

    # Downloads
    st.markdown(
        '<div class="section-title">Export your results</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2, gap="medium")

    with d1:
        st.download_button(
            "Download analysis report",
            data=report,
            file_name=report_path.name,
            mime="text/plain",
            use_container_width=True,
        )

    with d2:
        if heatmap_path.is_file():
            st.download_button(
                "Download Grad-CAM image",
                data=heatmap_path.read_bytes(),
                file_name=heatmap_path.name,
                mime="image/png",
                use_container_width=True,
            )

    with st.expander("View full technical report"):
        st.code(report, language="text")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## Plant AI")
    st.caption("Disease intelligence dashboard")
    st.divider()

    st.markdown("### Model information")
    st.markdown(
        "**Disease classification**  \n"
        "EfficientNetV2B0 · 10 classes  \n\n"
        "**Severity estimation**  \n"
        "EfficientNetV2B0 · 5 classes  \n\n"
        "**Explainability**  \n"
        "Grad-CAM"
    )

    st.divider()
    st.markdown("### Model performance")

    st.metric("10-class disease test accuracy", "94.81%")
    st.caption(
        "Measured on the project's held-out PlantVillage test split. "
        "Real-world performance may differ."
    )

    st.metric("Severity clean-split test accuracy", "68.41%")
    st.caption(
        "Measured on the separate severity dataset's clean test split."
    )

    st.divider()
    st.markdown("### Workflow")
    st.markdown(
        "1. Upload a tomato leaf image  \n"
        "2. Predict one of ten disease classes  \n"
        "3. Estimate severity  \n"
        "4. Inspect Grad-CAM  \n"
        "5. Review recommendations and download results"
    )

    st.divider()
    st.caption("Local academic prototype · No cloud inference")


# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>Plant Disease Intelligence</h1>
        <p>
            Analyze tomato leaves with 10-class disease classification,
            severity estimation, explainable AI and care recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Leaf image analysis</div>',
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    with st.container(border=True):
        st.subheader("Upload your image")

        uploaded = st.file_uploader(
            "Choose a tomato leaf image",
            type=["jpg", "jpeg", "png"],
            help="Choose a clear, well-lit image of a tomato leaf.",
            key="tomato_upload",
        )

        st.caption(
            "JPG, JPEG and PNG are supported. A single, clearly visible "
            "tomato leaf is recommended."
        )

        analyze_button = st.button(
            "Analyze leaf",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

with preview_col:
    with st.container(border=True):
        st.subheader("Image preview")

        if uploaded is not None:
            try:
                preview = Image.open(
                    io.BytesIO(uploaded.getvalue())
                ).convert("RGB")

                st.image(preview, use_container_width=True)
                st.caption(f"Selected image: {uploaded.name}")

            except Exception:
                st.error("This image could not be opened.")

        else:
            st.markdown(
                """
                <div style="
                    background:#F7F4FF;
                    border:1px dashed #B9A6E8;
                    border-radius:13px;
                    min-height:210px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                    text-align:center;
                    padding:20px;">
                    <div style="font-size:42px;">🍃</div>
                    <div style="color:#402779;font-weight:700;">
                        Your leaf image appears here
                    </div>
                    <div style="color:#625A73;font-size:13px;margin-top:8px;">
                        Upload an image to get started
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# ANALYSIS AND SESSION STATE
# ---------------------------------------------------------

if uploaded is not None:
    uploaded_bytes = uploaded.getvalue()
    signature = hashlib.sha256(uploaded_bytes).hexdigest()

    if st.session_state.get("upload_signature") != signature:
        st.session_state["upload_signature"] = signature
        st.session_state.pop("latest_report", None)
        st.session_state.pop("latest_report_path", None)
        st.session_state.pop("latest_heatmap_path", None)
        st.session_state.pop("latest_upload_signature", None)

if analyze_button and uploaded is not None:
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"leaf_{timestamp}_{unique_id}"

    image_path = UPLOADS / f"{stem}.jpg"
    report_path = RESULTS / f"{stem}_analysis.txt"
    heatmap_path = RESULTS / f"{stem}_full_analysis.png"

    try:
        preview = Image.open(
            io.BytesIO(uploaded.getvalue())
        ).convert("RGB")
        preview.save(image_path, format="JPEG", quality=95)

        analyze = load_pipeline()
        captured_output = io.StringIO()

        with st.spinner(
            "Loading models, predicting disease and severity, and generating Grad-CAM..."
        ):
            with contextlib.redirect_stdout(captured_output):
                analyze(str(image_path))

        if report_path.is_file():
            st.session_state["latest_report"] = report_path.read_text(
                encoding="utf-8"
            )
            st.session_state["latest_report_path"] = str(report_path)
            st.session_state["latest_heatmap_path"] = str(heatmap_path)
            st.session_state["latest_upload_signature"] = signature
        else:
            st.error("The analysis report was not created.")
            st.code(captured_output.getvalue() or "No technical output.")

    except Exception as exc:
        st.error("Analysis failed. See the details below.")
        st.exception(exc)

# Keep results visible when Streamlit reruns.
if st.session_state.get("latest_report"):
    if (
        st.session_state.get("latest_upload_signature")
        == st.session_state.get("upload_signature")
    ):
        st.divider()

        display_results(
            st.session_state["latest_report"],
            Path(st.session_state["latest_heatmap_path"]),
            Path(st.session_state["latest_report_path"]),
        )

st.markdown(
    """
    <div class="footer">
        TOMATO DISEASE INTELLIGENCE · MACHINE LEARNING PROJECT<br>
        Experimental predictions only. Confirm disease and treatment choices
        using reliable agricultural guidance.
    </div>
    """,
    unsafe_allow_html=True,
)
