"""DecodeClassify - a Copilot-inspired classification workspace."""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from src.data import FEATURES, dataset_summary, load_dataset
from src.etl import run_etl, run_uploaded_etl
from src.etl.extract import extract_titanic
from src.training import predict_passenger, train_classifier

ROOT = Path(__file__).resolve().parent
RAW_DATASET = ROOT / "data" / "raw" / "titanic_raw.csv"
PROCESSED_DATASET = ROOT / "data" / "processed" / "titanic_processed.csv"
ETL_REPORT = ROOT / "data" / "processed" / "etl_report.json"

st.set_page_config(page_title="DecodeClassify", page_icon="◆", layout="wide")

st.markdown(
    """
    <style>
    :root{--canvas:#0d1117;--panel:#161b22;--panel2:#0f141b;--border:#30363d;--muted:#8b949e;--text:#e6edf3;--purple:#a371f7;--green:#3fb950}
    .stApp{background:radial-gradient(circle at 68% -20%,#25134b 0,#0d1117 32%);color:var(--text)}
    .block-container{max-width:1440px;padding:1rem 2rem 6rem}
    [data-testid="stSidebar"]{background:#010409;border-right:1px solid var(--border)}
    [data-testid="stSidebar"]>div{padding-top:1.2rem}
    [data-testid="stHeader"]{background:transparent}
    .brand{font-size:1.08rem;font-weight:700;display:flex;gap:.65rem;align-items:center;margin-bottom:.2rem}
    .mark{display:grid;place-items:center;width:32px;height:32px;background:linear-gradient(135deg,#8957e5,#a371f7);border-radius:9px;box-shadow:0 0 24px #8957e555}
    .breadcrumb{font-size:.78rem;color:var(--muted);margin:.2rem 0 1rem}.breadcrumb b{color:#c9d1d9}
    .topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding:.2rem 0 .9rem;margin-bottom:1rem}
    .repo{font-weight:650}.repo span{color:var(--purple)}
    .branch{font-size:.75rem;padding:.3rem .6rem;border:1px solid var(--border);border-radius:999px;color:#b1bac4;background:#161b22}
    .hero{position:relative;overflow:hidden;border:1px solid var(--border);background:linear-gradient(140deg,#1c2330,#11151c 65%);border-radius:16px;padding:2rem 2.2rem;margin:.7rem 0 1rem}
    .hero:after{content:'';position:absolute;width:260px;height:260px;border-radius:50%;right:-80px;top:-110px;background:#8957e533;filter:blur(12px)}
    .hero h1{font-size:2.15rem;letter-spacing:-.035em;margin:0 0 .45rem}.hero p{max-width:720px;color:#b1bac4;line-height:1.6;margin:.2rem 0}
    .eyebrow{color:#bc8cff;text-transform:uppercase;letter-spacing:.12em;font-size:.68rem;font-weight:700}
    .pill{display:inline-block;border:1px solid #3d444d;border-radius:999px;padding:.3rem .62rem;margin:.8rem .35rem 0 0;color:#b1bac4;font-size:.75rem;background:#0d111799}
    .card{height:100%;border:1px solid var(--border);background:linear-gradient(145deg,#161b22,#11151b);border-radius:13px;padding:1.1rem 1.15rem}
    .card-icon{font-size:1.25rem;margin-bottom:.6rem}.card-title{font-weight:650;margin-bottom:.3rem}.card-copy{color:var(--muted);font-size:.82rem;line-height:1.5}
    .step{display:flex;gap:.8rem;padding:.8rem 0;border-bottom:1px solid #21262d}.step:last-child{border:none}.step-no{display:grid;place-items:center;min-width:26px;height:26px;border-radius:50%;background:#8957e522;border:1px solid #8957e566;color:#bc8cff;font-size:.75rem;font-weight:700}
    .step b{font-size:.88rem}.step small{display:block;color:var(--muted);margin-top:.15rem}
    .assistant{border:1px solid #8957e566;background:linear-gradient(145deg,#8957e512,#161b22);border-radius:13px;padding:1rem 1.1rem}
    .assistant-head{display:flex;gap:.5rem;align-items:center;color:#d8b9ff;font-weight:650;margin-bottom:.45rem}.assistant p{color:#b1bac4;font-size:.84rem;line-height:1.55;margin:.2rem 0}
    .status{border:1px solid var(--border);background:#161b22;padding:.75rem .8rem;border-radius:9px;color:#b1bac4;font-size:.76rem}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:.45rem}
    [data-testid="stMetric"]{background:#161b22;border:1px solid var(--border);padding:1rem;border-radius:12px}
    [data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden}
    [data-testid="stFileUploaderDropzone"]{background:#161b22;border:1px dashed #6e7681;border-radius:12px;padding:1rem}
    .prediction{border:1px solid #8957e5;background:#8957e512;border-radius:13px;padding:1.2rem 1.3rem;margin-top:1rem}
    .result-title{font-size:1.35rem;font-weight:700}.subtle{color:var(--muted);font-size:.82rem}
    div[data-testid="stButton"] button{border-radius:8px;font-weight:600}
    #MainMenu,footer{visibility:hidden}
    </style>
    """,
    unsafe_allow_html=True,
)


def topbar() -> None:
    st.markdown(
        '<div class="topbar"><div class="repo"><span>DecodeLabs</span> / 02-decode-classify</div><div class="branch">◆ main · local</div></div>',
        unsafe_allow_html=True,
    )


def assistant_note(title: str, copy: str) -> None:
    st.markdown(
        f'<div class="assistant"><div class="assistant-head">◆ Copilot insight · {title}</div><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="brand"><span class="mark">◆</span> DecodeClassify</div>', unsafe_allow_html=True)
    st.caption("AI classification workspace")
    st.markdown("---")
    page = st.radio(
        "Workspace",
        ["Overview", "Train model", "Predict", "Dataset"],
        captions=["Project health", "Run experiment", "Try an input", "Inspect rows"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("PROJECT 02 · DECODELABS")
    st.markdown('<div class="status"><span class="dot"></span>Local workspace<br><span class="subtle">No data leaves this device.</span></div>', unsafe_allow_html=True)

topbar()

uploaded = None
if RAW_DATASET.exists():
    uploaded = st.sidebar.file_uploader(
        "Replace Titanic raw source",
        type="csv",
        help="A valid upload replaces data/raw/titanic_raw.csv and regenerates the processed dataset.",
    )
if not RAW_DATASET.exists():
    empty_states = {
        "Overview": (
            "Initialize workspace",
            "Teach your first classifier.",
            "Connect the Kaggle Titanic training dataset, inspect its quality and train a supervised model in one guided workspace.",
            "Connect a dataset",
        ),
        "Train model": (
            "Experiment runner · locked",
            "Configure your first experiment.",
            "Connect labeled training data to unlock the split, model configuration, evaluation metrics and experiment results.",
            "Unlock model training",
        ),
        "Predict": (
            "Interactive inference · locked",
            "Ask the model a question.",
            "A trained pipeline is required before DecodeClassify can evaluate a new passenger and explain model confidence.",
            "Connect training data first",
        ),
        "Dataset": (
            "Data explorer · source required",
            "Inspect raw and processed data.",
            "Connect train.csv to compare the immutable Kaggle source, model-ready output and complete ETL lineage.",
            "Connect the raw source",
        ),
    }
    empty_eyebrow, empty_title, empty_copy, connect_title = empty_states[page]
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">{empty_eyebrow}</div>
          <h1>{empty_title}</h1>
          <p>{empty_copy}</p>
          <span class="pill">Binary classification</span><span class="pill">891 passengers</span><span class="pill">Local processing</span>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if ETL_REPORT.exists():
        previous_report = json.loads(ETL_REPORT.read_text(encoding="utf-8"))
        if previous_report.get("status") == "rejected":
            st.info("The previous ETL run did not produce a processed artifact. Connect a valid labeled CSV to retry.")
    intro, guide = st.columns([1.35, 1])
    with intro:
        st.subheader(connect_title)
        st.caption("Upload the official Kaggle train.csv. DecodeClassify validates the schema before enabling the model workspace.")
        uploaded = st.file_uploader("Drop Kaggle train.csv here", type="csv", label_visibility="collapsed")
        st.link_button(
            "Open Titanic dataset on Kaggle ↗",
            "https://www.kaggle.com/competitions/titanic/data?select=train.csv",
            use_container_width=True,
        )
        st.markdown("#### Expected schema")
        schema = pd.DataFrame(
            {
                "Field": ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
                "Role": ["Target", "Feature", "Feature", "Feature", "Feature", "Feature", "Feature", "Feature"],
                "Example": ["0 or 1", "1, 2, 3", "female", "28", "0", "0", "72.50", "C"],
            }
        )
        st.dataframe(schema, hide_index=True, use_container_width=True)
    with guide:
        st.markdown("#### Guided workflow")
        st.markdown(
            """
            <div class="card">
              <div class="step"><span class="step-no">1</span><div><b>Connect</b><small>Validate the Kaggle CSV and target.</small></div></div>
              <div class="step"><span class="step-no">2</span><div><b>Explore</b><small>Review balance, missing data and features.</small></div></div>
              <div class="step"><span class="step-no">3</span><div><b>Train</b><small>Split data and fit a simple classifier.</small></div></div>
              <div class="step"><span class="step-no">4</span><div><b>Explain</b><small>Read metrics and test a new passenger.</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        assistant_note("Why Titanic?", "It turns Project 2 into a real classification workflow: categorical data, missing values, a binary target and metrics that are easy to explain.")
    if uploaded is None:
        st.markdown("---")
        cols = st.columns(3)
        preview_cards = {
            "Overview": [
                ("◫", "Dataset aware", "Schema validation, missing-value profiling and class balance before training."),
                ("◇", "Reproducible", "A fixed random seed and stratified split make every experiment traceable."),
                ("◎", "Explainable", "Confusion matrix, precision, recall and F1 instead of an accuracy-only demo."),
            ],
            "Train model": [
                ("⚙", "Experiment settings", "Choose Logistic Regression or Decision Tree and configure the held-out test set."),
                ("◩", "Evaluation", "Accuracy, confusion matrix, precision, recall and F1 appear after training."),
                ("◆", "Copilot explanation", "Translate model results into clear supervised-learning concepts."),
            ],
            "Predict": [
                ("⌁", "Passenger input", "Enter class, age, fare, family counts, sex and embarkation port."),
                ("◎", "Class probability", "Return a predicted class with confidence from the trained pipeline."),
                ("◇", "Responsible context", "Explain that probability is not causality, fairness or a safety assessment."),
            ],
            "Dataset": [
                ("R", "Raw", "Preserve the original Kaggle upload without manual edits."),
                ("T", "Transform", "Normalize types, categories and missing values under an explicit contract."),
                ("L", "Processed", "Load a model-ready artifact with zero missing values and ETL lineage."),
            ],
        }
        cards = preview_cards[page]
        for column, (icon, title, copy) in zip(cols, cards):
            column.markdown(f'<div class="card"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-copy">{copy}</div></div>', unsafe_allow_html=True)
        st.stop()
try:
    if uploaded is not None:
        etl = run_uploaded_etl(uploaded.getvalue(), RAW_DATASET, PROCESSED_DATASET)
        st.success("ETL completed. Raw and processed artifacts are ready.")
    else:
        etl = run_etl(RAW_DATASET, PROCESSED_DATASET)
    data = load_dataset(PROCESSED_DATASET)
except Exception as error:
    st.error(f"ETL validation failed: {error}")
    if "Survived" in str(error):
        st.warning("This file has no usable target. Upload a labeled Titanic CSV with a binary Survived column.")
    st.stop()

for etl_warning in etl.warnings:
    st.warning(
        f"Dataset provenance: {etl_warning} Training is enabled for demonstration, "
        "but its evaluation metrics are not a ground-truth benchmark."
    )

summary = dataset_summary(data)
st.markdown(
    '<div class="breadcrumb">DecodeClassify / <b>workspace</b> / dataset connected</div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    st.markdown('<div class="eyebrow">Repository overview</div><h1>Model workspace</h1>', unsafe_allow_html=True)
    cols = st.columns(4)
    source_label = "Demo submission labels" if etl.warnings else "Kaggle train"
    cols[0].metric("Passengers", f"{summary['rows']:,}", source_label)
    cols[1].metric("Model features", summary["features"], "validated")
    cols[2].metric("Missing values", summary["missing"], f"{etl.missing_before} resolved by ETL")
    target_metric = "Proxy positive rate" if etl.warnings else "Survival rate"
    cols[3].metric(target_metric, f"{summary['survival_rate']:.1%}", "target balance")
    st.markdown("#### ETL execution")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Extracted", etl.raw_rows, "raw rows")
    e2.metric("Loaded", etl.processed_rows, "processed rows")
    e3.metric("Missing before", etl.missing_before, "raw quality")
    e4.metric("Missing after", etl.missing_after, "model-ready")
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Class distribution")
        distribution = data["Survived"].value_counts().sort_index().rename(index={0: "Did not survive", 1: "Survived"})
        st.bar_chart(distribution, color="#a371f7", height=310)
    with right:
        st.subheader("Workspace health")
        st.markdown(
            f"""
            <div class="card">
              <div class="step"><span class="step-no">✓</span><div><b>Schema valid</b><small>{len(FEATURES)} model features found.</small></div></div>
              <div class="step"><span class="step-no">✓</span><div><b>Target ready</b><small>Survived contains binary labels.</small></div></div>
              <div class="step"><span class="step-no">✓</span><div><b>ETL complete</b><small>{etl.missing_before} missing values resolved; {etl.missing_after} remain.</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        assistant_note("Read the balance", "Accuracy is useful, but not sufficient. After training, compare recall for both classes to see which outcome the model misses more often.")

elif page == "Train model":
    st.markdown('<div class="eyebrow">Experiment runner</div><h1>Train a classifier</h1>', unsafe_allow_html=True)
    settings, evaluation = st.columns([1, 1.75])
    with settings:
        st.markdown("#### Configuration")
        algorithm = st.selectbox("Algorithm", ["Logistic Regression", "Decision Tree"])
        test_percent = st.slider("Test set", 15, 35, 20, 5, format="%d%%")
        test_size = test_percent / 100
        depth = st.slider("Maximum tree depth", 2, 8, 4, disabled=algorithm != "Decision Tree")
        st.caption("Random state 42 · Stratified split · Missing-value imputation")
        run = st.button("▶ Run experiment", type="primary", use_container_width=True)
        assistant_note("Model choice", "Logistic Regression is the clearest baseline. Decision Tree is easier to visualize conceptually but can overfit when depth grows.")
    if run or "result" not in st.session_state:
        st.session_state.result = train_classifier(data, algorithm, test_size, depth)
        st.session_state.algorithm = algorithm
    result = st.session_state.result
    with evaluation:
        st.markdown("#### Evaluation")
        a, b, c = st.columns(3)
        a.metric("Accuracy", f"{result.accuracy:.1%}")
        b.metric("Training rows", result.train_rows)
        c.metric("Test rows", result.test_rows)
        tabs = st.tabs(["Confusion matrix", "Class metrics", "Copilot explanation"])
        matrix = pd.DataFrame(result.confusion, index=["Actual: no", "Actual: yes"], columns=["Predicted: no", "Predicted: yes"])
        tabs[0].dataframe(matrix, use_container_width=True)
        metrics = pd.DataFrame(result.report).T.loc[["0", "1"], ["precision", "recall", "f1-score", "support"]]
        tabs[1].dataframe(metrics.style.format({"precision":"{:.2f}","recall":"{:.2f}","f1-score":"{:.2f}","support":"{:.0f}"}), use_container_width=True)
        with tabs[2]:
            assistant_note("What happened?", f"The model classified {result.accuracy:.1%} of the held-out passengers correctly. Use the confusion matrix to inspect errors instead of treating accuracy as the whole story.")

elif page == "Predict":
    if "result" not in st.session_state:
        st.session_state.result = train_classifier(data)
    st.markdown('<div class="eyebrow">Interactive inference</div><h1>Ask the model</h1>', unsafe_allow_html=True)
    form, context = st.columns([1.45, 1])
    with form:
        with st.form("passenger"):
            c1, c2, c3 = st.columns(3)
            passenger = {
                "Pclass": c1.selectbox("Passenger class", [1, 2, 3]),
                "Sex": c2.selectbox("Sex", ["female", "male"]),
                "Age": c3.number_input("Age", 0.1, 100.0, 30.0),
                "SibSp": c1.number_input("Siblings / spouses", 0, 8, 0),
                "Parch": c2.number_input("Parents / children", 0, 6, 0),
                "Fare": c3.number_input("Fare", 0.0, 600.0, 32.0),
                "Embarked": c1.selectbox("Embarked", ["S", "C", "Q"]),
            }
            submitted = st.form_submit_button("Ask DecodeClassify", type="primary", use_container_width=True)
        if submitted:
            prediction, probability = predict_passenger(st.session_state.result, passenger)
            label = "Survived" if prediction else "Did not survive"
            st.markdown(f'<div class="prediction"><div class="eyebrow">Model response</div><div class="result-title">{label}</div><p class="subtle">Confidence: {probability:.1%}</p></div>', unsafe_allow_html=True)
    with context:
        st.markdown("#### Prediction context")
        st.markdown('<div class="card"><div class="card-title">Educational classification</div><div class="card-copy">The result is based on patterns in a small historical dataset. It is not a causal explanation, safety assessment or judgment about a real person.</div></div>', unsafe_allow_html=True)
        assistant_note("Interpret confidence", "Confidence describes the model's probability for its selected class. It does not measure whether the historical data is fair or complete.")

else:
    st.markdown('<div class="eyebrow">Data explorer</div><h1>Inspect the source</h1>', unsafe_allow_html=True)
    raw_data = extract_titanic(RAW_DATASET)
    raw_tab, processed_tab, lineage_tab = st.tabs(["Raw", "Processed", "ETL lineage"])
    with raw_tab:
        st.caption("Immutable source copied from the Kaggle upload.")
        st.dataframe(raw_data.head(100), use_container_width=True, height=460)
    with processed_tab:
        selected_class = st.selectbox("Filter survival", ["All", "Survived", "Did not survive"])
        filtered = data
        if selected_class == "Survived":
            filtered = data[data["Survived"] == 1]
        elif selected_class == "Did not survive":
            filtered = data[data["Survived"] == 0]
        st.dataframe(filtered.head(100), use_container_width=True, height=460)
    with lineage_tab:
        lineage = pd.DataFrame(
            [
                ["Extract", "data/raw/titanic_raw.csv", etl.raw_rows, etl.missing_before],
                ["Transform", "Select, normalize, impute, validate", etl.processed_rows, etl.missing_after],
                ["Load", "data/processed/titanic_processed.csv", etl.processed_rows, etl.missing_after],
            ],
            columns=["Stage", "Artifact / operation", "Rows", "Missing values"],
        )
        st.dataframe(lineage, hide_index=True, use_container_width=True)
    assistant_note("Privacy and provenance", "The dataset is processed locally and attributed to its Kaggle source. Historical sensitive attributes are used only for this supervised-learning exercise.")
