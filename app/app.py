%%writefile app.py

import streamlit as st
import tempfile
import pandas as pd
import cv2

from backend.pcb_backend import process_pcb

st.set_page_config(page_title="PCB Defect Detection", layout="wide")


st.markdown("""
<style>

/* HEADER */
.header {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(90deg, #00ffe0, #7a5cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 20px;
}

/* REMOVE DEFAULT HR */
hr {
    display: none;
}

/*  CARD WITH CONTROLLED NEON GLOW */
.card {
    padding: 15px;
    border-radius: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,200,0.4);

    /*  IMPORTANT: tight glow (no bars) */
    box-shadow:
        0 0 6px rgba(0,255,200,0.3),
        0 0 12px rgba(0,255,200,0.15);

    transition: 0.3s ease;
}

/*  Hover glow effect */
.card:hover {
    box-shadow:
        0 0 10px rgba(0,255,200,0.6),
        0 0 20px rgba(0,255,200,0.25);
}

/* RESULT */
.result {
    padding: 10px;
    border-radius: 8px;
    margin: 6px 0;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,200,0.4);
}

/* CONFIDENCE */
.conf {
    color: #00ffe0;
    font-weight: bold;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #00ffe0, #7a5cff);
    color: black;
    border-radius: 8px;
}

/* OPTIONAL: glow for uploader box */
[data-testid="stFileUploaderDropzone"] {
    border: 1px solid rgba(0,255,200,0.4) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 8px rgba(0,255,200,0.2);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# HEADER
# ------------------------------
st.markdown('<div class="header">🔍 PCB Defect Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning + Computer Vision Inspection System</div>', unsafe_allow_html=True)

# ------------------------------
# UPLOAD
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Template Image")
    template = st.file_uploader("Upload Template", type=["jpg","png"], key="template")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Test Image")
    test = st.file_uploader("Upload Test", type=["jpg","png"], key="test")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------
# PROCESS
# ------------------------------
if template and test:

    template_file = tempfile.NamedTemporaryFile(delete=False)
    template_file.write(template.read())

    test_file = tempfile.NamedTemporaryFile(delete=False)
    test_file.write(test.read())

    with st.spinner(" AI analyzing PCB..."):
        output, results = process_pcb(template_file.name, test_file.name)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1.2, 1])

    # IMAGE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🖼️ Detection Output")
        st.image(output_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # RESULTS
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Detected Defects")

        if len(results) == 0:
            st.warning("No defects detected")
        else:
            for r in results:
                st.markdown(
                    f'<div class="result">✔ {r[0]} | <span class="conf">{r[1]:.2f}</span></div>',
                    unsafe_allow_html=True
                )

        st.markdown(f"### 🔢 Total Defects: {len(results)}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # DOWNLOAD
    cv2.imwrite("result.png", output)

    col1, col2 = st.columns(2)

    with col1:
        with open("result.png", "rb") as f:
            st.download_button("⬇ Download Image", f, "result.png")

    if results:
        df = pd.DataFrame(results, columns=["Defect","Confidence"])
        csv = df.to_csv(index=False).encode()

        with col2:
            st.download_button("⬇ Download Report", csv, "report.csv")
