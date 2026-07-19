"""DecodeClassify: an interactive Iris + KNN supervised-learning lab."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, SPECIES, TARGET, load_dataset
from src.etl import run_etl
from src.etl.extract import extract_iris
from src.training import predict_flower, train_classifier, tune_neighbors

ROOT = Path(__file__).resolve().parent
RAW_DATASET = ROOT / "data" / "raw" / "iris.csv"
PROCESSED_DATASET = ROOT / "data" / "processed" / "iris_processed.csv"

FEATURE_LABELS = {
    "sepal_length_cm": "Sepal length",
    "sepal_width_cm": "Sepal width",
    "petal_length_cm": "Petal length",
    "petal_width_cm": "Petal width",
}

st.set_page_config(
    page_title="DecodeClassify · Iris KNN Lab",
    page_icon="✦",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root{
      --canvas:#07111f;--panel:#0d1b2a;--panel-2:#10243a;--border:#25435f;
      --muted:#8fa7bd;--text:#edf7ff;--cyan:#2dd4bf;--blue:#60a5fa;
      --orange:#ff7a3d;--green:#5ee89b;
    }
    .stApp{
      color:var(--text);
      background:
        radial-gradient(circle at 82% -10%,#123b5d 0,transparent 32%),
        linear-gradient(#ffffff08 1px,transparent 1px),
        linear-gradient(90deg,#ffffff08 1px,transparent 1px),
        var(--canvas);
      background-size:auto,30px 30px,30px 30px,auto;
    }
    .block-container{max-width:1420px;padding:1.1rem 2rem 5rem}
    [data-testid="stSidebar"]{background:#06101c;border-right:1px solid var(--border)}
    [data-testid="stSidebar"]>div{padding-top:1.2rem}
    [data-testid="stHeader"]{background:transparent}
    .brand{display:flex;align-items:center;gap:.7rem;font-weight:760;font-size:1.05rem}
    .brand-mark{
      display:grid;place-items:center;width:34px;height:34px;border-radius:10px;
      color:#031019;background:linear-gradient(135deg,var(--cyan),var(--blue));
      box-shadow:0 0 26px #2dd4bf44;
    }
    .topbar{
      display:flex;justify-content:space-between;align-items:center;gap:1rem;
      padding:.2rem 0 .9rem;margin-bottom:1.1rem;border-bottom:1px solid var(--border);
    }
    .repo{font-weight:680}.repo span{color:var(--cyan)}
    .badge{
      border:1px solid #2dd4bf66;background:#2dd4bf12;color:#9ff7e8;
      border-radius:999px;padding:.28rem .65rem;font-size:.74rem;
    }
    .hero{
      position:relative;overflow:hidden;padding:2rem 2.15rem;margin:.35rem 0 1.1rem;
      border:1px solid var(--border);border-radius:18px;
      background:linear-gradient(135deg,#11283d 0,#0b1826 67%);
      box-shadow:0 20px 70px #00000035;
    }
    .hero:after{
      content:"K = 5";position:absolute;right:4%;top:7%;font-size:5rem;font-weight:850;
      letter-spacing:-.08em;color:#2dd4bf0d;transform:rotate(-4deg);
    }
    .hero h1{font-size:2.35rem;letter-spacing:-.045em;margin:.15rem 0 .5rem}
    .hero p{max-width:780px;color:#b4c8d9;line-height:1.65;margin:0}
    .eyebrow{
      color:var(--cyan);text-transform:uppercase;letter-spacing:.14em;
      font-size:.68rem;font-weight:800;
    }
    .pill{
      display:inline-block;margin:.95rem .38rem 0 0;padding:.3rem .62rem;
      border:1px solid #36546d;border-radius:999px;background:#07111f99;
      color:#bcd0df;font-size:.74rem;
    }
    .card{
      height:100%;padding:1.05rem 1.1rem;border:1px solid var(--border);
      border-radius:14px;background:linear-gradient(145deg,#10243a,#0a1725);
    }
    .card-kicker{color:var(--orange);font-size:.67rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}
    .card-title{font-size:1rem;font-weight:720;margin:.32rem 0}
    .card-copy{color:var(--muted);font-size:.82rem;line-height:1.55}
    .step{display:flex;gap:.8rem;padding:.78rem 0;border-bottom:1px solid #1d354a}
    .step:last-child{border-bottom:0}
    .step-no{
      display:grid;place-items:center;min-width:28px;height:28px;border-radius:8px;
      background:#2dd4bf14;border:1px solid #2dd4bf66;color:#8ff3e3;
      font-size:.72rem;font-weight:800;
    }
    .step b{font-size:.87rem}.step small{display:block;color:var(--muted);margin-top:.12rem;line-height:1.4}
    .insight{
      margin-top:.75rem;padding:.9rem 1rem;border:1px solid #ff7a3d55;
      border-radius:12px;background:linear-gradient(135deg,#ff7a3d10,#10243a);
    }
    .insight b{color:#ffb18d;font-size:.82rem}.insight p{color:#aebfd0;font-size:.8rem;line-height:1.5;margin:.3rem 0 0}
    .prediction{
      padding:1.25rem 1.35rem;border:1px solid #2dd4bf99;border-radius:15px;
      background:linear-gradient(135deg,#2dd4bf18,#10243a);margin-top:1rem;
    }
    .prediction h2{margin:.2rem 0 .15rem;text-transform:capitalize}
    .status{
      border:1px solid var(--border);border-radius:10px;background:#0d1b2a;
      padding:.75rem .8rem;color:#a9bdce;font-size:.76rem;
    }
    .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:.45rem}
    .subtle{color:var(--muted);font-size:.79rem}
    [data-testid="stMetric"]{
      border:1px solid var(--border);border-radius:13px;
      background:linear-gradient(145deg,#10243a,#0a1725);padding:.9rem 1rem;
    }
    [data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:12px;overflow:hidden}
    div[data-testid="stButton"] button,div[data-testid="stFormSubmitButton"] button{
      border-radius:9px;font-weight:700;
    }
    #MainMenu,footer{visibility:hidden}
    @media(max-width:800px){
      .block-container{padding:1rem 1rem 4rem}.hero{padding:1.5rem}.hero h1{font-size:1.85rem}
      .hero:after{display:none}.topbar{align-items:flex-start;flex-direction:column}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def insight(title: str, copy: str) -> None:
    """Render a consistent educational callout."""
    st.markdown(
        f'<div class="insight"><b>Blueprint note · {escape(title)}</b>'
        f"<p>{escape(copy)}</p></div>",
        unsafe_allow_html=True,
    )


def card(kicker: str, title: str, copy: str) -> str:
    """Return safe card markup for constant educational content."""
    return (
        '<div class="card">'
        f'<div class="card-kicker">{escape(kicker)}</div>'
        f'<div class="card-title">{escape(title)}</div>'
        f'<div class="card-copy">{escape(copy)}</div>'
        "</div>"
    )


@st.cache_data(show_spinner=False)
def prepare_dataset() -> tuple[object, pd.DataFrame]:
    """Materialize the processed dataset once per source version."""
    etl_result = run_etl(RAW_DATASET, PROCESSED_DATASET)
    return etl_result, load_dataset(PROCESSED_DATASET)


try:
    etl, data = prepare_dataset()
except Exception as error:
    st.error(f"Dataset initialization failed: {error}")
    st.info("Restore data/raw/iris.csv or provide the required Iris schema.")
    st.stop()

dataset_signature = (
    len(data),
    tuple(data[TARGET].value_counts().sort_index().items()),
    float(data[list(FEATURES)].sum().sum()),
)
if (
    st.session_state.get("dataset_signature") != dataset_signature
    or "model_result" not in st.session_state
    or "model_config" not in st.session_state
):
    st.session_state.dataset_signature = dataset_signature
    st.session_state.model_result = None
    st.session_state.model_config = (5, 0.2)
    st.session_state.tuning_result = None
    st.session_state.tuning_signature = None

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>'
        "<span>DecodeClassify</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Iris · KNN learning workspace")
    st.markdown("---")
    page = st.radio(
        "Workspace",
        ["Overview", "Train model", "Predict", "Dataset"],
        captions=["Understand the system", "Run an experiment", "Classify a flower", "Inspect the evidence"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("PROJECT 02 · DECODELABS")
    st.markdown(
        '<div class="status"><span class="dot"></span>Dataset ready'
        '<br><span class="subtle">150 local samples · no network</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="topbar"><div class="repo"><span>DecodeLabs</span> / '
    '02-decode-classify</div><div class="badge">Local · reproducible pipeline</div></div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Project 2 · Data Classification Using AI</div>
          <h1>From four measurements to one species.</h1>
          <p>Explore a complete supervised-learning pipeline built around the
          Iris benchmark: understand the input, preserve a held-out test set,
          scale features without leakage, train KNN, and validate its output.</p>
          <span class="pill">150 samples</span>
          <span class="pill">3 balanced classes</span>
          <span class="pill">4 dimensions</span>
          <span class="pill">K-Nearest Neighbors</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("Samples", f"{len(data)}", "50 per species")
    metrics[1].metric("Features", f"{len(FEATURES)}", "centimeters")
    metrics[2].metric("Classes", f"{data[TARGET].nunique()}", "balanced")
    metrics[3].metric("Missing values", f"{int(data.isna().sum().sum())}", "validated")

    visual, architecture = st.columns([1.55, 1], gap="large")
    with visual:
        st.subheader("Petal measurements reveal the pattern")
        st.scatter_chart(
            data,
            x="petal_length_cm",
            y="petal_width_cm",
            color=TARGET,
            size=55,
            height=360,
        )
        st.caption(
            "Each point is a labeled example. KNN predicts from the classes of "
            "the closest scaled examples, not from handwritten if/else rules."
        )
    with architecture:
        st.subheader("Input → process → output")
        st.markdown(
            """
            <div class="card">
              <div class="step"><span class="step-no">01</span><div><b>Load</b><small>Validate 150 labeled Iris observations.</small></div></div>
              <div class="step"><span class="step-no">02</span><div><b>Split</b><small>Shuffle and stratify training vs. held-out test data.</small></div></div>
              <div class="step"><span class="step-no">03</span><div><b>Scale + fit</b><small>Learn scaling only from training rows, then fit KNN.</small></div></div>
              <div class="step"><span class="step-no">04</span><div><b>Validate</b><small>Review confusion, precision, recall, and F1.</small></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        insight(
            "Why scaling matters",
            "KNN uses distance. Standardization prevents a wider numeric range "
            "from dominating which flowers count as nearest.",
        )

    st.markdown("#### Requirement coverage")
    coverage = st.columns(4)
    items = [
        ("INPUT", "Iris benchmark", "150 samples, 3 species, and the four measurements named in the PDF."),
        ("PROCESS", "80/20 split", "Shuffled, stratified, and reproducible with random state 42."),
        ("MODEL", "Scaled KNN", "StandardScaler and KNeighborsClassifier live in one pipeline."),
        ("OUTPUT", "Deep validation", "Accuracy plus confusion matrix and multiclass macro F1."),
    ]
    for column, (kicker, title, copy) in zip(coverage, items):
        column.markdown(card(kicker, title, copy), unsafe_allow_html=True)

elif page == "Train model":
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Experiment runner</div>
          <h1>Train and validate KNN.</h1>
          <p>Choose the number of neighbors and the held-out test size.
          The same reproducible split makes experiments comparable.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    settings, evaluation = st.columns([1, 1.8], gap="large")
    with settings:
        st.subheader("Configuration")
        neighbors = st.slider(
            "Neighbors (K)",
            min_value=1,
            max_value=15,
            value=int(st.session_state.model_config[0]),
            step=1,
            help="Small K can overfit; large K can smooth away useful local structure.",
        )
        test_percent = st.slider(
            "Held-out test set",
            min_value=20,
            max_value=30,
            value=int(st.session_state.model_config[1] * 100),
            step=5,
            format="%d%%",
        )
        run_experiment = st.button(
            "▶ Run experiment",
            type="primary",
            use_container_width=True,
        )
        if run_experiment:
            with st.spinner("Splitting, scaling, training, and validating…"):
                st.session_state.model_result = train_classifier(
                    data,
                    neighbors=neighbors,
                    test_size=test_percent / 100,
                )
                st.session_state.model_config = (neighbors, test_percent / 100)
            st.success("Experiment completed with a fresh held-out evaluation.")
        elif (neighbors, test_percent / 100) != st.session_state.model_config:
            st.info("Configuration changed. Run the experiment to apply it.")
        st.caption("Shuffle on · Stratified split · random_state = 42")
        insight(
            "Test-set integrity",
            "The scaler is fitted only on training rows. The test set remains "
            "unseen until final evaluation.",
        )

    result = st.session_state.model_result
    active_k, active_test_size = st.session_state.model_config
    if result is None:
        with evaluation:
            st.subheader("Evaluation")
            st.info(
                "No experiment has run in this session. Choose K and the test "
                "size, then select **Run experiment**."
            )
            st.markdown(
                card(
                    "READY",
                    "The dataset is validated",
                    "Training starts only when you request it, so every visible "
                    "result has a clear configuration and fresh evaluation.",
                ),
                unsafe_allow_html=True,
            )
        st.stop()
    with evaluation:
        st.subheader(f"Evaluation · K={active_k} · {active_test_size:.0%} test")
        score_columns = st.columns(4)
        score_columns[0].metric("Accuracy", f"{result.accuracy:.1%}")
        score_columns[1].metric("Macro F1", f"{result.macro_f1:.1%}")
        score_columns[2].metric("Training rows", result.train_rows)
        score_columns[3].metric("Test rows", result.test_rows)

        confusion_tab, metrics_tab, tuning_tab, explain_tab = st.tabs(
            ["Confusion matrix", "Class metrics", "Tune K", "How it works"]
        )
        with confusion_tab:
            matrix = pd.DataFrame(
                result.confusion,
                index=[f"Actual · {name}" for name in result.class_names],
                columns=[f"Predicted · {name}" for name in result.class_names],
            )
            st.dataframe(matrix, use_container_width=True)
            st.caption(
                "Diagonal cells are correct predictions; off-diagonal cells "
                "show which species the model confused."
            )
            diagnostics = []
            total = sum(sum(row) for row in result.confusion)
            for class_index, class_name in enumerate(result.class_names):
                true_positive = result.confusion[class_index][class_index]
                false_negative = sum(result.confusion[class_index]) - true_positive
                false_positive = (
                    sum(row[class_index] for row in result.confusion) - true_positive
                )
                true_negative = (
                    total - true_positive - false_negative - false_positive
                )
                diagnostics.append(
                    {
                        "Species (one-vs-rest)": class_name,
                        "TP": true_positive,
                        "FP": false_positive,
                        "FN": false_negative,
                        "TN": true_negative,
                    }
                )
            st.markdown("##### One-vs-rest diagnostics")
            st.dataframe(
                pd.DataFrame(diagnostics),
                hide_index=True,
                use_container_width=True,
            )
        with metrics_tab:
            report = pd.DataFrame(result.report).T
            class_rows = [name for name in result.class_names if name in report.index]
            class_metrics = report.loc[
                class_rows, ["precision", "recall", "f1-score", "support"]
            ]
            st.dataframe(
                class_metrics.style.format(
                    {
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                        "f1-score": "{:.3f}",
                        "support": "{:.0f}",
                    }
                ),
                use_container_width=True,
            )
            st.caption(
                "F1 is the harmonic mean of precision and recall. Macro F1 gives "
                "every species equal weight, even if future data is imbalanced."
            )
        with tuning_tab:
            tuning_signature = (dataset_signature, active_test_size)
            st.caption(
                "Compare K=1…15 with five-fold cross-validation on training "
                "rows only. This is intentionally separate from final evaluation."
            )
            if st.button(
                "Analyze K values",
                key="analyze_k_values",
                use_container_width=True,
            ):
                with st.spinner("Cross-validating candidate K values…"):
                    tuning = tune_neighbors(
                        data,
                        max_neighbors=15,
                        folds=5,
                        test_size=active_test_size,
                    )
                    st.session_state.tuning_result = tuning.assign(
                        error_rate=1 - tuning["mean_accuracy"]
                    )
                    st.session_state.tuning_signature = tuning_signature
            if st.session_state.get("tuning_signature") != tuning_signature:
                st.session_state.tuning_result = None
            tuning = st.session_state.get("tuning_result")
            if tuning is None:
                st.info("Select **Analyze K values** to generate the error curve.")
            else:
                recommended = int(
                    tuning.loc[tuning["error_rate"].idxmin(), "neighbors"]
                )
                st.line_chart(
                    tuning.set_index("neighbors")[["error_rate"]],
                    height=280,
                    color="#ff7a3d",
                )
                st.caption(
                    f"Minimum cross-validation error occurs at K={recommended}. "
                    "The held-out test rows stay unseen."
                )
                st.dataframe(
                    tuning.style.format(
                        {
                            "mean_accuracy": "{:.3f}",
                            "std_accuracy": "{:.3f}",
                            "error_rate": "{:.3f}",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
        with explain_tab:
            st.markdown(
                f"""
                <div class="card">
                  <div class="step"><span class="step-no">1</span><div><b>Shuffle + stratify</b><small>{result.train_rows} training rows and {result.test_rows} held-out rows.</small></div></div>
                  <div class="step"><span class="step-no">2</span><div><b>Standardize</b><small>Training means become 0 and variances become 1.</small></div></div>
                  <div class="step"><span class="step-no">3</span><div><b>Find neighbors</b><small>The {active_k} closest scaled flowers vote for a species.</small></div></div>
                  <div class="step"><span class="step-no">4</span><div><b>Validate once</b><small>Predictions are compared with labels that were not used to fit the model.</small></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "Predict":
    result = st.session_state.model_result
    active_k, _ = st.session_state.model_config
    if result is None:
        st.markdown(
            """
            <section class="hero">
              <div class="eyebrow">Interactive inference · model required</div>
              <h1>Train before you predict.</h1>
              <p>Open <strong>Train model</strong>, choose K, and run an
              experiment. This keeps every prediction tied to a visible,
              validated model configuration.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.warning("Prediction is locked until an experiment runs in this session.")
        st.stop()
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Interactive inference · Active model K={active_k}</div>
          <h1>Classify a new flower.</h1>
          <p>Enter the four measurements used by the Iris benchmark. The model
          applies its training scaler before comparing the new point with its
          nearest labeled examples.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    form_column, context_column = st.columns([1.5, 1], gap="large")
    with form_column:
        with st.form("iris_measurements"):
            left, right = st.columns(2)
            flower = {
                "sepal_length_cm": left.number_input(
                    "Sepal length (cm)", 0.1, 10.0, 5.1, 0.1
                ),
                "sepal_width_cm": right.number_input(
                    "Sepal width (cm)", 0.1, 10.0, 3.5, 0.1
                ),
                "petal_length_cm": left.number_input(
                    "Petal length (cm)", 0.1, 10.0, 1.4, 0.1
                ),
                "petal_width_cm": right.number_input(
                    "Petal width (cm)", 0.1, 10.0, 0.2, 0.1
                ),
            }
            submitted = st.form_submit_button(
                "Classify flower",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            species, confidence, probabilities = predict_flower(result, flower)
            st.markdown(
                '<div class="prediction"><div class="eyebrow">Model output</div>'
                f"<h2>Iris {escape(species)}</h2>"
                f'<div class="subtle">Neighbor-vote confidence · {confidence:.1%}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            probability_frame = pd.DataFrame(
                {
                    "species": list(probabilities.keys()),
                    "probability": list(probabilities.values()),
                }
            ).set_index("species")
            st.bar_chart(
                probability_frame,
                y="probability",
                height=250,
                color="#2dd4bf",
            )
    with context_column:
        st.subheader("What the model sees")
        values = pd.DataFrame(
            {
                "Measurement": [FEATURE_LABELS[feature] for feature in FEATURES],
                "Value (cm)": [flower[feature] for feature in FEATURES],
            }
        )
        st.dataframe(values, hide_index=True, use_container_width=True)
        st.markdown(
            card(
                "MODEL CONTRACT",
                "Four numbers in, one class out",
                "The pipeline expects centimeters in the same order and meaning "
                "as the training dataset. A prediction outside those ranges is "
                "an extrapolation, not proof.",
            ),
            unsafe_allow_html=True,
        )
        insight(
            "Interpret confidence",
            "For KNN, this probability is the share of neighbors voting for a "
            "class. It is not a universal probability of botanical truth.",
        )

else:
    raw_data = extract_iris(RAW_DATASET)
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Evidence workspace</div>
          <h1>Inspect the Iris benchmark.</h1>
          <p>Trace the local source through validation and transformation, then
          see why standardization changes distance without changing the class.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    dataset_metrics = st.columns(4)
    dataset_metrics[0].metric("Raw rows", etl.raw_rows)
    dataset_metrics[1].metric("Processed rows", etl.processed_rows)
    dataset_metrics[2].metric("Duplicates removed", etl.duplicates_removed)
    dataset_metrics[3].metric("Missing after", etl.missing_after)

    raw_tab, processed_tab, schema_tab, scaling_tab = st.tabs(
        ["Raw source", "Processed", "Schema", "Scaling"]
    )
    with raw_tab:
        st.caption("Versioned local input · data/raw/iris.csv")
        st.dataframe(raw_data, hide_index=True, use_container_width=True, height=440)
    with processed_tab:
        selected_species = st.selectbox(
            "Filter species",
            ["All", *SPECIES],
        )
        filtered = data
        if selected_species != "All":
            filtered = data[data[TARGET] == selected_species]
        st.dataframe(filtered, hide_index=True, use_container_width=True, height=400)
        st.download_button(
            "Download processed CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="iris_processed.csv",
            mime="text/csv",
        )
    with schema_tab:
        ranges = data[list(FEATURES)].agg(["min", "max", "mean"]).T.reset_index()
        ranges.columns = ["Feature", "Minimum", "Maximum", "Mean"]
        ranges["Feature"] = ranges["Feature"].map(FEATURE_LABELS)
        st.dataframe(
            ranges.style.format(
                {"Minimum": "{:.1f}", "Maximum": "{:.1f}", "Mean": "{:.2f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        counts = (
            data[TARGET]
            .value_counts()
            .reindex(SPECIES)
            .rename_axis("Species")
            .rename("Samples")
            .reset_index()
        )
        st.dataframe(counts, hide_index=True, use_container_width=True)
    with scaling_tab:
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(data[list(FEATURES)]),
            columns=list(FEATURES),
        )
        comparison = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS[feature] for feature in FEATURES],
                "Raw mean": data[list(FEATURES)].mean().values,
                "Raw std": data[list(FEATURES)].std(ddof=0).values,
                "Scaled mean": scaled.mean().values,
                "Scaled std": scaled.std(ddof=0).values,
            }
        )
        st.dataframe(
            comparison.style.format(
                {
                    "Raw mean": "{:.3f}",
                    "Raw std": "{:.3f}",
                    "Scaled mean": "{:.3f}",
                    "Scaled std": "{:.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        insight(
            "Demonstration vs. training",
            "This table fits a scaler to the full dataset only to illustrate "
            "mean 0 and standard deviation 1. The actual model fits its scaler "
            "exclusively on the training split.",
        )
