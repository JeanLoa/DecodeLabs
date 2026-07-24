"""DecodeVision: an inspectable OCR recognition workspace."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.vision import (
    OCREngineUnavailable,
    PreprocessingConfig,
    TesseractEngine,
    decode_image,
    draw_word_boxes,
    preprocess_image,
    run_ocr_pipeline,
)
from src.vision.validation import evaluate_milestone

ROOT = Path(__file__).resolve().parent
SAMPLE_IMAGE = ROOT / "data" / "samples" / "decodevision_invoice.png"

PSM_OPTIONS = {
    "Automatic layout": 3,
    "Uniform text block": 6,
    "Single text line": 7,
    "Sparse text": 11,
}


st.set_page_config(
    page_title="DecodeVision · OCR Inspection Lab",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root{
      --canvas:#08101b;--panel:#0e1a28;--panel-2:#111f30;--line:#26394b;
      --ink:#edf4f8;--muted:#8ea0b1;--orange:#ff6b3d;--blue:#4ea7ff;
      --green:#39d98a;--red:#ff6b6b;--cream:#f2eadf;
    }
    .stApp{
      background:
        linear-gradient(#ffffff05 1px,transparent 1px),
        linear-gradient(90deg,#ffffff05 1px,transparent 1px),
        radial-gradient(circle at 86% -12%,#173e62 0,transparent 31%),
        var(--canvas);
      background-size:32px 32px,32px 32px,auto,auto;color:var(--ink);
    }
    [data-testid="stHeader"]{background:transparent}
    [data-testid="stSidebar"]{background:#07111c;border-right:1px solid var(--line)}
    [data-testid="stSidebar"]>div{padding-top:1.2rem}
    .block-container{max-width:1460px;padding:1.1rem 2rem 6rem}
    .brand{display:flex;align-items:center;gap:.75rem;font-size:1.08rem;font-weight:780}
    .brand-mark{
      display:grid;place-items:center;width:38px;height:38px;border-radius:11px;
      color:#07111c;background:linear-gradient(135deg,var(--orange),#ffb45c);
      box-shadow:0 0 28px #ff6b3d42;font-weight:900;
    }
    .brand-sub{color:var(--muted);font-size:.73rem;margin:.38rem 0 1.2rem 3.15rem}
    .topbar{
      display:flex;justify-content:space-between;align-items:center;gap:1rem;
      min-height:46px;padding-bottom:.8rem;border-bottom:1px solid var(--line);
      color:var(--muted);font-size:.78rem;
    }
    .topbar strong{color:var(--ink);font-size:.9rem}
    .top-badge{
      display:inline-flex;align-items:center;gap:.45rem;padding:.34rem .72rem;
      border:1px solid var(--line);border-radius:999px;background:#0e1a28;
    }
    .signal{width:7px;height:7px;border-radius:50%;background:var(--green)}
    .signal.off{background:var(--red)}
    .hero{
      position:relative;overflow:hidden;margin:1.1rem 0;padding:2rem 2.1rem;
      border:1px solid var(--line);border-radius:20px;
      background:linear-gradient(135deg,#13283c,#0c1724 68%);
      box-shadow:0 24px 70px #0000002e;
    }
    .hero:after{
      content:"OCR";position:absolute;right:3%;top:-18%;font-size:11rem;font-weight:900;
      letter-spacing:-.08em;color:#4ea7ff0a;
    }
    .eyebrow{
      color:var(--orange);font-size:.67rem;font-weight:850;letter-spacing:.15em;
      text-transform:uppercase;
    }
    .hero h1{font-size:2.45rem;letter-spacing:-.045em;margin:.35rem 0 .55rem}
    .hero p{max-width:800px;color:#b1c0ce;line-height:1.65;margin:0}
    .hero-pills{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}
    .hero-pill{
      padding:.28rem .62rem;border:1px solid #36506a;border-radius:999px;
      background:#07111f99;color:#bfd0de;font-size:.7rem;
    }
    .panel{
      border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,#102033,#0c1724);
      padding:1.05rem 1.1rem;height:100%;
    }
    .panel-label{
      color:var(--orange);font-size:.65rem;font-weight:850;letter-spacing:.12em;
      text-transform:uppercase;margin-bottom:.35rem;
    }
    .panel-title{font-weight:760;margin-bottom:.25rem}
    .panel-copy{color:var(--muted);font-size:.79rem;line-height:1.55}
    .status-card{
      border:1px solid var(--line);border-radius:13px;background:#0e1a28;
      padding:.8rem .85rem;color:#afbfcc;font-size:.76rem;line-height:1.5;
    }
    .metric-card{
      border:1px solid var(--line);border-radius:15px;background:#0e1a28;
      padding:.9rem 1rem;height:100%;
    }
    .metric-label{color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.08em}
    .metric-value{font-size:1.55rem;font-weight:820;margin:.2rem 0}
    .metric-note{color:var(--muted);font-size:.71rem}
    .pass{color:var(--green)}.fail{color:var(--red)}.accent{color:var(--orange)}
    .result-box{
      border:1px solid #39d98a66;border-radius:16px;background:linear-gradient(135deg,#39d98a12,#0e1a28);
      padding:1rem 1.1rem;white-space:pre-wrap;min-height:150px;color:#e9fff5;
      font-family:"Cascadia Code","Consolas",monospace;font-size:.84rem;line-height:1.65;
    }
    .empty-result{
      border:1px dashed #3b5063;border-radius:16px;padding:2rem;text-align:center;
      color:var(--muted);background:#0b1622;
    }
    .step{
      display:flex;gap:.85rem;padding:.85rem 0;border-bottom:1px solid #213446;
    }
    .step:last-child{border-bottom:0}
    .step-no{
      display:grid;place-items:center;flex:0 0 auto;width:32px;height:32px;
      border:1px solid #ff6b3d66;border-radius:9px;background:#ff6b3d12;
      color:#ff9a77;font-size:.72rem;font-weight:850;
    }
    .step b{font-size:.88rem}.step small{display:block;color:var(--muted);line-height:1.45;margin-top:.15rem}
    .gate{
      display:flex;align-items:flex-start;gap:.75rem;border:1px solid var(--line);
      border-radius:14px;background:#0e1a28;padding:.9rem;margin:.55rem 0;
    }
    .gate-icon{
      display:grid;place-items:center;flex:0 0 auto;width:28px;height:28px;
      border-radius:8px;background:#39d98a18;color:var(--green);font-weight:900;
    }
    .gate-icon.fail{background:#ff6b6b17;color:var(--red)}
    .gate b{font-size:.83rem}.gate p{margin:.2rem 0 0;color:var(--muted);font-size:.74rem;line-height:1.45}
    [data-testid="stMetric"]{
      border:1px solid var(--line);border-radius:14px;background:#0e1a28;padding:.8rem .9rem;
    }
    [data-testid="stFileUploader"]{
      border:1px dashed #3c5368;border-radius:14px;background:#0b1622;padding:.35rem;
    }
    [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}
    [data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line)}
    [data-testid="stTabs"] button{font-weight:700}
    div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{
      border-radius:9px;font-weight:750;
    }
    div[data-testid="stButton"] button[kind="primary"]{
      background:var(--orange);border-color:var(--orange);color:#07111c;
    }
    #MainMenu,footer{visibility:hidden}
    @media(max-width:800px){
      .block-container{padding:1rem 1rem 5rem}.hero{padding:1.5rem}.hero h1{font-size:1.85rem}
      .hero:after{display:none}.topbar{align-items:flex-start;flex-direction:column}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def stage_panel(label: str, title: str, copy: str) -> str:
    return (
        '<div class="panel">'
        f'<div class="panel-label">{escape(label)}</div>'
        f'<div class="panel-title">{escape(title)}</div>'
        f'<div class="panel-copy">{escape(copy)}</div>'
        "</div>"
    )


def engine_help() -> None:
    st.error(
        "The OpenCV pipeline is ready, but native Tesseract OCR was not found. "
        "Install it once and restart the app to enable recognition."
    )
    st.code(
        "winget install -e --id UB-Mannheim.TesseractOCR\n"
        '$env:TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"',
        language="powershell",
    )


engine = TesseractEngine()
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "result_signature" not in st.session_state:
    st.session_state.result_signature = None

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">◉</span>'
        "<span>DecodeVision</span></div>"
        '<div class="brand-sub">OCR inspection workspace</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Workspace",
        ["Recognition lab", "Pipeline anatomy", "Validation gate"],
        captions=[
            "Extract and inspect text",
            "Understand every transformation",
            "Audit the Project 4 milestone",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("NATIVE ENGINE")
    state_class = "" if engine.available else " off"
    state_copy = "Tesseract ready" if engine.available else "Tesseract setup required"
    st.markdown(
        f'<div class="status-card"><span class="signal{state_class}" '
        'style="display:inline-block;margin-right:.45rem"></span>'
        f"<b>{state_copy}</b><br>"
        "OpenCV preprocessing remains available offline.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("PROJECT 04 · DECODELABS")
    st.caption("Local processing · no uploaded data leaves this device")

status_class = "" if engine.available else " off"
st.markdown(
    '<div class="topbar"><strong>DecodeLabs / 04-decodevision-ocr</strong>'
    f'<span class="top-badge"><span class="signal{status_class}"></span>'
    f"{'Recognition engine online' if engine.available else 'Preprocessing mode'}</span></div>",
    unsafe_allow_html=True,
)

if page == "Recognition lab":
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Project 4 · Image or Text Recognition</div>
          <h1>Turn pixels into validated text.</h1>
          <p>Inspect a complete optical character recognition pipeline:
          decode the visual matrix, remove noise, force a binary decision,
          correct rotation, run Tesseract, and keep only words that pass a
          visible confidence gate.</p>
          <div class="hero-pills">
            <span class="hero-pill">OpenCV preprocessing</span>
            <span class="hero-pill">Tesseract OCR</span>
            <span class="hero-pill">80% confidence gate</span>
            <span class="hero-pill">Word-level evidence</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    controls, workspace = st.columns([0.82, 2.18], gap="large")
    with controls:
        st.subheader("Mission controls")
        source = st.radio(
            "Input source",
            ["Built-in sample", "Upload image"],
            horizontal=True,
        )
        uploaded = None
        if source == "Upload image":
            uploaded = st.file_uploader(
                "PNG or JPG document",
                type=["png", "jpg", "jpeg"],
            )
        threshold_mode_label = st.selectbox(
            "Thresholding",
            ["Adaptive Gaussian", "Otsu global"],
            help="Adaptive thresholding performs better with uneven lighting.",
        )
        psm_label = st.selectbox(
            "Page segmentation",
            list(PSM_OPTIONS),
            index=1,
        )
        confidence = st.slider(
            "Confidence gate",
            min_value=50,
            max_value=99,
            value=80,
            help="The Project 4 milestone requires at least 80%.",
        )
        blur = st.select_slider(
            "Gaussian blur kernel",
            options=[1, 3, 5, 7],
            value=3,
        )
        deskew_enabled = st.toggle("Automatic deskew", value=True)
        execute = st.button(
            "Execute OCR pipeline",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Recognition is deterministic for the same image, parameters, "
            "Tesseract version and language data."
        )

    image_bytes: bytes | None
    if source == "Built-in sample":
        image_bytes = SAMPLE_IMAGE.read_bytes() if SAMPLE_IMAGE.exists() else None
    else:
        image_bytes = uploaded.getvalue() if uploaded is not None else None

    with workspace:
        if image_bytes is None:
            st.markdown(
                '<div class="empty-result">Upload a document image to initialize '
                "the visual pipeline.</div>",
                unsafe_allow_html=True,
            )
        else:
            try:
                image_bgr = decode_image(image_bytes)
                config = PreprocessingConfig(
                    blur_kernel=blur,
                    threshold_mode=(
                        "adaptive"
                        if threshold_mode_label == "Adaptive Gaussian"
                        else "otsu"
                    ),
                    deskew=deskew_enabled,
                )
                preview = preprocess_image(image_bgr, config)
            except ValueError as error:
                st.error(str(error))
                st.stop()

            height, width = image_bgr.shape[:2]
            metrics = st.columns(4)
            metrics[0].metric("Width", f"{width:,} px")
            metrics[1].metric("Height", f"{height:,} px")
            metrics[2].metric("Channels", "3 → 1")
            metrics[3].metric("Deskew angle", f"{preview.detected_angle:+.2f}°")

            tabs = st.tabs(["Original", "Grayscale", "Blurred", "Binary", "Deskewed"])
            images = [
                preview.original_rgb,
                preview.grayscale,
                preview.blurred,
                preview.binary,
                preview.deskewed,
            ]
            captions = [
                "RGB visual input",
                "One intensity channel",
                f"Gaussian noise reduction · {blur}×{blur}",
                f"{threshold_mode_label} threshold",
                "OCR-ready horizontal baseline",
            ]
            for tab, image, caption in zip(tabs, images, captions, strict=True):
                with tab:
                    st.image(image, caption=caption, use_column_width=True)

            signature = (
                len(image_bytes),
                sum(image_bytes[:5000]),
                threshold_mode_label,
                psm_label,
                confidence,
                blur,
                deskew_enabled,
            )
            if execute:
                if not engine.available:
                    engine_help()
                else:
                    try:
                        with st.spinner("Reading the visual matrix…"):
                            st.session_state.pipeline_result = run_ocr_pipeline(
                                image_bgr,
                                preprocessing_config=config,
                                confidence_threshold=float(confidence),
                                page_segmentation_mode=PSM_OPTIONS[psm_label],
                                engine=engine,
                            )
                            st.session_state.result_signature = signature
                    except OCREngineUnavailable as error:
                        st.error(str(error))
                        engine_help()

            result = st.session_state.pipeline_result
            if result is not None and st.session_state.result_signature == signature:
                recognition = result.recognition
                st.subheader("Recognition output")
                result_metrics = st.columns(4)
                result_metrics[0].metric(
                    "Validated confidence",
                    f"{recognition.validated_confidence:.1f}%",
                    "PASS" if recognition.passes_confidence_gate else "BELOW GATE",
                )
                result_metrics[1].metric(
                    "Accepted words",
                    len(recognition.accepted_words),
                    f"{recognition.accepted_ratio:.0%} coverage",
                )
                result_metrics[2].metric(
                    "All detected words",
                    len(recognition.words),
                )
                result_metrics[3].metric(
                    "PSM",
                    recognition.page_segmentation_mode,
                    psm_label,
                )

                output_columns = st.columns([1.1, 1], gap="large")
                with output_columns[0]:
                    st.markdown("**Validated text**")
                    final_text = recognition.validated_text or "No word passed the active gate."
                    st.markdown(
                        f'<div class="result-box">{escape(final_text)}</div>',
                        unsafe_allow_html=True,
                    )
                    download_columns = st.columns(2)
                    download_columns[0].download_button(
                        "Download TXT",
                        recognition.validated_text,
                        file_name="decodevision-validated.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                    download_columns[1].download_button(
                        "Download JSON",
                        json.dumps(recognition.to_dict(), indent=2, ensure_ascii=False),
                        file_name="decodevision-evidence.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with output_columns[1]:
                    st.markdown("**Bounding-box evidence**")
                    st.image(
                        draw_word_boxes(result),
                        caption="Green ≥ gate · red < gate",
                        use_column_width=True,
                    )

                with st.expander("Inspect every recognized word"):
                    rows: list[dict[str, Any]] = [
                        {
                            "Text": word.text,
                            "Confidence": word.confidence,
                            "Decision": (
                                "Accepted"
                                if word.confidence >= recognition.confidence_threshold
                                else "Rejected"
                            ),
                            "X": word.x,
                            "Y": word.y,
                            "Width": word.width,
                            "Height": word.height,
                        }
                        for word in recognition.words
                    ]
                    st.dataframe(
                        rows,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%",
                            )
                        },
                    )
            elif result is not None:
                st.info("Parameters changed. Execute the pipeline again to refresh the evidence.")

elif page == "Pipeline anatomy":
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Input → Process → Output</div>
          <h1>Deconstruct the machine’s optic nerve.</h1>
          <p>An image begins as a three-dimensional array. DecodeVision reduces
          irrelevant variation before the OCR engine interprets shapes as text.
          Every transformation remains visible and independently testable.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    stage_columns = st.columns(4, gap="medium")
    stage_content = [
        ("01 · INGEST", "Decode the matrix", "Validate PNG/JPG bytes and materialize height × width × RGB channels."),
        ("02 · REDUCE", "Grayscale + blur", "Collapse color and smooth micro-noise that can fragment character contours."),
        ("03 · SEPARATE", "Threshold + deskew", "Force foreground/background contrast and align text to a horizontal baseline."),
        ("04 · INTERPRET", "OCR + confidence", "Ask Tesseract for words, coordinates and confidence, then enforce the 80% gate."),
    ]
    for column, content in zip(stage_columns, stage_content, strict=True):
        with column:
            st.markdown(stage_panel(*content), unsafe_allow_html=True)

    architecture, modes = st.columns([1.25, 1], gap="large")
    with architecture:
        st.subheader("Systematic preprocessing")
        steps = [
            ("01", "Grayscale conversion", "RGB becomes one intensity channel; color no longer distracts recognition."),
            ("02", "Gaussian blur", "A configurable odd kernel suppresses small artifacts before segmentation."),
            ("03", "Binary decision", "Adaptive Gaussian or Otsu thresholding produces explicit foreground contours."),
            ("04", "Deskew", "Foreground geometry estimates rotation and returns text to its baseline."),
            ("05", "Tesseract data", "The engine returns text, confidence, line identity and bounding-box coordinates."),
            ("06", "Confidence filter", "Only words at or above the selected gate become validated output."),
        ]
        for number, title, copy in steps:
            st.markdown(
                f'<div class="step"><span class="step-no">{number}</span><div>'
                f"<b>{escape(title)}</b><small>{escape(copy)}</small></div></div>",
                unsafe_allow_html=True,
            )
    with modes:
        st.subheader("Page segmentation modes")
        st.markdown(
            """
            | PSM | Use case |
            |---:|---|
            | `3` | Automatic detection for varied layouts |
            | `6` | One uniform block of document text |
            | `7` | A single text line or plate |
            | `11` | Sparse labels scattered across an image |
            """
        )
        st.info(
            "Cosmetic resizing is never used to fabricate accuracy. The OCR "
            "engine receives the actual thresholded matrix shown in the lab."
        )

else:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">The Gatekeeper Rule</div>
          <h1>Prove the pipeline, not just the output.</h1>
          <p>Project 4 defines four uncompromising validations: native-library
          integration, demonstrable preprocessing, confidence of at least 80%,
          and pristine visual confirmation.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    current = st.session_state.pipeline_result
    if current is None:
        st.markdown(
            '<div class="empty-result">Run a recognition experiment first. '
            "Its evidence will be audited here.</div>",
            unsafe_allow_html=True,
        )
        static_checks = [
            ("Library integration", "Tesseract must return structured OCR data."),
            ("Preprocessing integrity", "Grayscale and threshold stages must be inspectable."),
            ("Accuracy benchmarking", "Validated output must reach at least 80% confidence."),
            ("Visual confirmation", "Text and bounding boxes must be legible and spatially grounded."),
        ]
        for index, (title, copy) in enumerate(static_checks, start=1):
            st.markdown(
                f'<div class="gate"><span class="gate-icon fail">{index}</span><div>'
                f"<b>{escape(title)}</b><p>{escape(copy)}</p></div></div>",
                unsafe_allow_html=True,
            )
    else:
        checks = evaluate_milestone(current)
        passed = sum(check.passed for check in checks)
        headline_class = "pass" if passed == len(checks) else "fail"
        summary_columns = st.columns(3)
        summary_columns[0].metric("Checks passed", f"{passed}/{len(checks)}")
        summary_columns[1].metric(
            "Confidence",
            f"{current.recognition.validated_confidence:.1f}%",
        )
        summary_columns[2].metric(
            "Milestone",
            "READY" if passed == len(checks) else "REVIEW",
        )
        st.markdown(
            f'<h3 class="{headline_class}">'
            f'{"Milestone validated" if passed == len(checks) else "Evidence needs attention"}'
            "</h3>",
            unsafe_allow_html=True,
        )
        for check in checks:
            icon_class = "" if check.passed else " fail"
            st.markdown(
                f'<div class="gate"><span class="gate-icon{icon_class}">'
                f'{"✓" if check.passed else "!"}</span><div>'
                f"<b>{escape(check.name)}</b><p>{escape(check.evidence)}</p></div></div>",
                unsafe_allow_html=True,
            )

        st.subheader("Audit contract")
        st.json(
            {
                "project": "DecodeVision",
                "pipeline": "OpenCV preprocessing → Tesseract OCR",
                "milestone_passed": passed == len(checks),
                "checks": [
                    {
                        "name": check.name,
                        "passed": check.passed,
                        "evidence": check.evidence,
                    }
                    for check in checks
                ],
            },
            expanded=False,
        )
