import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from changeledger import compare_images

def Image_to_png_bytes(array):
    """Encode an image array as PNG bytes for downloads."""
    buffer = io.BytesIO()
    image = np.asarray(array).astype(np.uint8)

    if image.ndim == 2:
        Image.fromarray(image, mode="L").save(buffer, format="PNG")
    else:
        Image.fromarray(image).save(buffer, format="PNG")

    return buffer.getvalue()




# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ChangeLedger | Visual Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --page: #f5f5f0;
        --surface: #ffffff;
        --ink: #202820;
        --muted: #59645c;
        --line: #d9dfd6;
        --green: #245c43;
        --green-hover: #194832;
        --green-soft: #e9f1eb;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    h1 { font-size: 2.5rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.65rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.15rem !important; font-weight: 700 !important; }

    p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--ink);
    }

    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
        font-size: 0.88rem;
    }

    [data-testid="stSidebar"] {
        background: #e9eee7;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: #24372b;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #203c2c !important;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(32, 40, 32, 0.035);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 600;
        font-size: 0.85rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    [data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        background: #fafbf8;
        border: 1px dashed #aab8a9;
        border-radius: 9px;
    }

    [data-testid="stFileUploader"] button {
        color: #234b35 !important;
        border-color: #9eafa0 !important;
        background: #ffffff !important;
    }

    div[data-testid="stImage"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 650;
        transition: background 0.15s ease;
    }

    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background: var(--green) !important;
        color: #ffffff !important;
        border: 1px solid var(--green) !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background: var(--green-hover) !important;
        border-color: var(--green-hover) !important;
        color: #ffffff !important;
    }

    div.stButton > button:not([kind="primary"]),
    div.stDownloadButton > button:not([kind="primary"]) {
        background: #ffffff !important;
        color: #263b2c !important;
        border: 1px solid #cbd5ca !important;
    }

    [data-testid="stTabs"] button {
        color: #59645c !important;
        font-weight: 650;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #245c43 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #245c43 !important;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary p {
        color: var(--ink) !important;
        font-weight: 600;
    }

    input, textarea {
        color: #202820 !important;
        background: #ffffff !important;
    }

    [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #cbd5ca;
    }

    hr {
        border-color: var(--line);
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    code, pre {
        color: #253c2c !important;
        background: #edf1e9 !important;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Header
# --------------------------------------------------

st.markdown('<div class="cl-brand">◈ ChangeLedger / Visual Intelligence</div>',
            unsafe_allow_html=True)

st.title("See what changed.")
st.markdown(
    '<div class="cl-subtitle">'
    'Compare two images, locate candidate changes, and generate an inspection report.'
    '</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar: settings
# --------------------------------------------------

with st.sidebar:
    st.markdown("## Workspace")
    st.caption("BASELINE ENGINE · v0.1.0")
    st.divider()

    st.markdown("### Detection settings")

    threshold = st.slider(
        "Difference sensitivity",
        min_value=5,
        max_value=100,
        value=25,
        help="Lower values detect smaller pixel differences but may add noise.",
    )

    min_area = st.number_input(
        "Minimum region area (pixels)",
        min_value=1,
        max_value=100000,
        value=100,
        step=25,
    )

    min_area_fraction = st.number_input(
        "Minimum region area fraction",
        min_value=0.0,
        max_value=0.1,
        value=0.0005,
        step=0.0005,
        format="%.4f",
        help="Small regions below this fraction of the image are filtered out.",
    )

    max_regions = st.number_input(
        "Maximum regions",
        min_value=1,
        max_value=500,
        value=100,
        step=10,
    )

    st.divider()
    st.caption("METHOD")
    st.markdown("ORB alignment + pixel differences")
    st.caption("Research-informed classical baseline")
    st.caption("Not a semantic object detector")


# --------------------------------------------------
# Upload workspace
# --------------------------------------------------

left_upload, right_upload = st.columns(2, gap="large")

with left_upload:
    st.markdown("### 01 / Before")
    before_file = st.file_uploader(
        "Upload the original image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="before_upload",
    )
    if before_file:
        st.image(before_file.getvalue(), use_container_width=True)

with right_upload:
    st.markdown("### 02 / After")
    after_file = st.file_uploader(
        "Upload the newer image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="after_upload",
    )
    if after_file:
        st.image(after_file.getvalue(), use_container_width=True)


st.divider()

ready = before_file is not None and after_file is not None

if not ready:
    st.info("Upload both images to start a comparison.")
    
compare_clicked = st.button(
    "↗  Compare images",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)


# --------------------------------------------------
# Run the engine
# --------------------------------------------------

if compare_clicked:
    try:
        before_bytes = before_file.getvalue()
        after_bytes = after_file.getvalue()

        with st.spinner("Aligning images and analyzing candidate changes..."):
            result = compare_images(
                before_bytes,
                after_bytes,
                threshold=int(threshold),
                min_area=int(min_area),
                min_area_fraction=float(min_area_fraction),
                max_regions=int(max_regions),
            )

        st.session_state["comparison_result"] = result
        st.session_state["comparison_settings"] = {
            "threshold": int(threshold),
            "min_area": int(min_area),
            "min_area_fraction": float(min_area_fraction),
            "max_regions": int(max_regions),
        }

    except Exception as exc:
        st.error(f"Comparison failed: {exc}")
        st.caption("Check that both uploads contain valid images and try again.")


# --------------------------------------------------
# Results workspace
# --------------------------------------------------

result = st.session_state.get("comparison_result")

if result is not None:
    report = result["report"]
    images = result["images"]

    st.divider()
    st.markdown("## Comparison results")
    st.caption("Candidate changes are generated by the classical baseline engine.")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Candidate regions",
        report.get("candidate_region_count", 0),
    )

    metric2.metric(
        "Changed pixels",
        f'{report.get("changed_pixel_fraction", 0) * 100:.2f}%',
    )

    metric3.metric(
        "Valid image area",
        f'{report.get("valid_pixel_fraction", 0) * 100:.1f}%',
    )

    alignment = report.get("alignment", {})
    metric4.metric(
        "Alignment inliers",
        alignment.get("homography_inliers", 0),
    )

    # Main comparison tabs
    tab_compare, tab_regions, tab_diagnostics, tab_report = st.tabs(
        [
            "Visual comparison",
            "Changed regions",
            "Alignment diagnostics",
            "Export report",
        ]
    )

    with tab_compare:
        view = st.radio(
            "Choose visualization",
            options=[
                "Annotated image",
                "Overlay",
                "Difference heatmap",
                "Change mask",
                "Aligned after image",
            ],
            horizontal=True,
        )

        image_lookup = {
            "Annotated image": "annotated",
            "Overlay": "overlay",
            "Difference heatmap": "difference_heatmap",
            "Change mask": "change_mask",
            "Aligned after image": "after_aligned",
        }

        selected_key = image_lookup[view]
        st.image(
            images[selected_key],
            caption=view,
            use_container_width=True,
        )

        st.caption(
            "The heatmap and mask represent pixel differences, not confirmed "
            "real-world object changes."
        )

        image_buffer = io.BytesIO()
        output_image = Image.fromarray(
            np.asarray(images[selected_key]).astype(np.uint8)
        )
        output_image.save(image_buffer, format="PNG")

        st.download_button(
            "Download current visualization",
            data=image_buffer.getvalue(),
            file_name=f"changeledger_{selected_key}.png",
            mime="image/png",
        )

    with tab_regions:
        regions = report.get("regions", [])

        if not regions:
            st.info("No candidate regions passed the current detection filters.")
        else:
            st.markdown(f"**{len(regions)} candidate regions detected**")

            for region in regions:
                region_id = region.get("id", "?")
                bbox = region.get("bbox", [])
                area = region.get("area", region.get("pixel_area", 0))
                fraction = region.get("area_fraction", 0)
                centroid = region.get("centroid", [])

                with st.expander(f"Region {region_id}"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.write("Bounding box `[x, y, width, height]`")
                        st.code(str(bbox))
                        st.write("Area in pixels")
                        st.write(area)

                    with col_b:
                        st.write("Image area fraction")
                        st.write(f"{fraction * 100:.3f}%")
                        st.write("Centroid")
                        st.code(str(centroid))

            st.caption(
                "Bounding boxes indicate candidate regions. They do not establish "
                "whether a change is meaningful or what object changed."
            )

    with tab_diagnostics:
        st.markdown("### Image alignment")

        st.json(alignment)

        st.markdown("### Detection summary")
        st.json({
            "method": report.get("method"),
            "image_size": report.get("image_size"),
            "changed_pixel_count": report.get("changed_pixel_count"),
            "changed_pixel_fraction": report.get("changed_pixel_fraction"),
            "valid_pixel_fraction": report.get("valid_pixel_fraction"),
            "parameters": report.get("parameters"),
            "limitations": report.get("limitations"),
        })

    with tab_report:
        st.markdown("### Inspection report")
        st.caption("Download the structured report for further review or integration.")

        report_json = json.dumps(report, indent=2, ensure_ascii=False)

        st.download_button(
            "Download JSON report",
            data=report_json.encode("utf-8"),
            file_name="changeledger_report.json",
            mime="application/json",
            type="primary",
        )

        st.download_button(
            "Download annotated image",
            data=Image_to_png_bytes(images["annotated"]),
            file_name="changeledger_annotated.png",
            mime="image/png",
        )

        with st.expander("Preview raw JSON"):
            st.code(report_json, language="json")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "CHANGELEDGER · MILESTONE 2 · Classical computer-vision baseline"
)
