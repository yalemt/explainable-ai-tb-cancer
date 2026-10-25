"""TB & Cancer CT research prototype (MSc Data Science thesis, AAU).
Run:  streamlit run app.py
It shows CT slices, reads a clinical narrative with transparent rules and, only if a trained model package is
present, produces probabilities with Grad-CAM explanations. Without a model it never invents a prediction."""
import os, hashlib, datetime
import numpy as np, pandas as pd, streamlit as st
from core import ct_io, model_io, explain
from core.narrative import analyze_narrative

st.set_page_config(page_title="TB & Cancer CT research prototype", page_icon="🫁", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.environ.get("TBC_MODEL_DIR", os.path.join(BASE, "model_package"))
AUDIT_PATH = os.path.join(BASE, "audit_log.csv")

st.error("**Research prototype, not a medical device.** Do not use it to diagnose, treat or decide care for any patient. "
         "Outputs describe how a research model behaves on de-identified data; every result needs review by a qualified clinician.")
st.title("🫁 TB and cancer CT: research prototype")
st.caption("Explainable AI for interpreting CT imaging and clinical narratives in tuberculosis and cancer characterization · MSc Data Science, Addis Ababa University")

@st.cache_resource(show_spinner=False)
def get_package(path):
    return model_io.load_package(path)

@st.cache_data(show_spinner=False)
def parse_upload(payload, organ, k):
    names = [n.lower() for n, _ in payload]
    blobs = [b for _, b in payload]
    if all(n.endswith((".png", ".jpg", ".jpeg")) for n in names):
        sl = ct_io.load_image_slices(blobs)
        return ct_io.to_rgb_slices(sl, k=k, already_windowed=True), {"source": "image slices", "n_slices": len(sl)}
    vol, info = ct_io.load_dicom_series(blobs)
    info["source"] = "DICOM series"
    return ct_io.to_rgb_slices(vol, organ_system=organ, k=k), info

pkg = get_package(PKG_DIR)
card = pkg["card"] or {}
classes = card.get("classes", model_io.CLASSES_DEFAULT)

with st.sidebar:
    st.header("Status")
    if pkg["ct_model"] is not None:
        st.success(f"CT model loaded: {card.get('name', 'unnamed')} v{card.get('version', '?')}")
    else:
        st.warning("No trained CT model. The app runs in **inspection mode**: you can view slices and analyze a narrative, but it will not predict a disease.")
        for m in pkg["messages"]:
            st.caption(m)
    st.header("Data protection")
    ok = st.checkbox("I confirm the data are de-identified and I am authorised to process them in this environment.")
    audit_on = st.checkbox("Keep a local audit log (no images, no names)", value=False)
    st.caption("This app runs on your own computer and makes no network calls. Nothing is uploaded anywhere.")

if not ok:
    st.info("Tick the confirmation in the sidebar to begin.")
    st.stop()

tab_ct, tab_txt, tab_res, tab_exp, tab_about = st.tabs(["1 · CT input", "2 · Narrative", "3 · Result", "4 · Explanation", "5 · About this model"])

# ---------------------------------------------------------------- 1 CT
with tab_ct:
    c1, c2, c3 = st.columns([3, 1, 1])
    files = c1.file_uploader("DICOM series (many files) or PNG/JPG slices", accept_multiple_files=True, type=None)
    organ = c2.selectbox("Body region", list(ct_io.WINDOWS), help="Sets the CT window. A lung window is wrong for the abdomen.")
    k = c3.slider("Slices used", 4, 32, int(card.get("n_slices", 16)))
    slices, info = None, None
    if files:
        try:
            slices, info = parse_upload(tuple((f.name, f.getvalue()) for f in files), organ, k)
        except Exception as e:
            st.error(f"Could not read the upload: {e}")
    if slices is not None:
        if info.get("header_has_identifiers"):
            st.warning("These DICOM headers still contain patient identifiers. The app does not display them, but de-identify the files before sharing them.")
        st.write({k_: v for k_, v in info.items() if k_ != "header_has_identifiers"})
        cols = st.columns(4)
        for i, s in enumerate(slices[:8]):
            cols[i % 4].image(s, caption=f"slice {i + 1}")
        st.session_state["slices"], st.session_state["organ"] = slices, organ
    else:
        st.info("Upload a scan to see its slices.")

# ---------------------------------------------------------------- 2 Narrative
with tab_txt:
    a, b = st.columns(2)
    exam = a.text_input("Type of examination", placeholder="e.g. chest, triphasic CT")
    text = b.text_area("Clinical data / narrative", placeholder="e.g. r/o active TB", height=80)
    if exam or text:
        r = analyze_narrative(exam, text)
        st.session_state["narr"] = r
        st.markdown("**What the transparent rules see**")
        st.dataframe(pd.DataFrame({"field": list(r.keys()), "value": [str(v) for v in r.values()]}), hide_index=True)
        st.caption("This is the same rule set the thesis uses for labelling. It reads the words; it does not know the patient. "
                   "'Masked' text hides diagnosis words so you can see what remains.")
    else:
        st.info("Enter a narrative to see how it is read.")

# ---------------------------------------------------------------- 3 Result
with tab_res:
    if pkg["ct_model"] is None:
        st.info("Prediction needs a trained model package. Train the CT model in Part B of the Colab notebook, export the package (last cell) "
                "and copy it into the `model_package` folder. Until then the app will not produce a diagnosis-like output.")
    elif "slices" not in st.session_state:
        st.info("Upload a scan in step 1.")
    elif st.button("Run analysis", type="primary"):
        with st.spinner("Analyzing slices…"):
            ct_p, per_slice = model_io.predict_ct(pkg["ct_model"], st.session_state["slices"], int(card.get("image_size", 224)))
        probs = {"CT only": ct_p}
        used_text = False
        if pkg["text_model"] is not None and "narr" in st.session_state:
            tx_p = model_io.text_probs(pkg["text_model"], st.session_state["narr"]["text_masked"], classes)
            probs["Narrative only (masked)"] = tx_p
            probs["Fused"] = model_io.fuse(ct_p, tx_p, float(card.get("fusion_weight_text", 0.3)))
            used_text = True
        st.session_state["per_slice"] = per_slice
        df_p = pd.DataFrame(probs, index=classes)
        st.bar_chart(df_p)
        final = probs["Fused"] if used_text else ct_p
        d = model_io.decide(final, classes, float(card.get("abstain_threshold", 0.60)))
        (st.warning if d["needs_human_review"] else st.success)(d["message"])
        st.caption("Why 'Indeterminate' is routed to a person: the thesis treats uncertain findings as a result in their own right, not as an error.")
        if audit_on:
            row = {"time_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"), "model": card.get("version", "?"),
                   "n_slices": len(st.session_state["slices"]), "top": d["top"], "confidence": round(d["confidence"], 3),
                   "needs_review": d["needs_human_review"], "slice0_hash": hashlib.sha1(st.session_state["slices"][0].tobytes()).hexdigest()[:10]}
            pd.DataFrame([row]).to_csv(AUDIT_PATH, mode="a", header=not os.path.exists(AUDIT_PATH), index=False)
        st.dataframe(df_p.style.format("{:.1%}"))
    m = card.get("metrics")
    if m:
        st.markdown("**How this model performed when it was tested** (from the model card)")
        st.json(m)
    elif pkg["ct_model"] is not None:
        st.warning("The model card has no test results. Treat the output as uncalibrated.")

# ---------------------------------------------------------------- 4 Explanation
with tab_exp:
    if pkg["ct_model"] is None or "per_slice" not in st.session_state:
        st.info("Run an analysis first (step 3).")
    else:
        import torch
        from PIL import Image
        from torchvision import transforms
        size = int(card.get("image_size", 224))
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                 transforms.Normalize(model_io.IMAGENET_MEAN, model_io.IMAGENET_STD)])
        ps = st.session_state["per_slice"]; sl = st.session_state["slices"]
        top_class = int(ps.mean(0).argmax())
        order = np.argsort(-ps[:, top_class])[:3]
        st.write(f"Heat maps for the three slices that support '{classes[top_class]}' most. Bright areas mean the model weighted that region more.")
        cols = st.columns(3)
        for col, i in zip(cols, order):
            x = tf(Image.fromarray(sl[i]))
            cam, _ = explain.gradcam(pkg["ct_model"], x, top_class)
            col.image(explain.overlay(sl[i], cam, size), caption=f"slice {i + 1}")
        st.caption("A heat map shows where the model looked, not that it reasoned correctly. Check it against the anatomy, and look at the wrong cases too.")
        if st.button("Run the sanity check (weight randomization)"):
            rho = explain.randomization_check(pkg["ct_model"], tf(Image.fromarray(sl[order[0]])))
            (st.error if rho > 0.7 else st.success)(f"Correlation between the real map and a map from a scrambled model: {rho:.2f}. "
                      + ("High: the map barely depends on what was learned, so do not rely on it." if rho > 0.7 else "Low: the map depends on the learned weights."))
        if pkg["text_model"] is not None and "narr" in st.session_state:
            st.markdown("**Words moving the narrative model** (masked text)")
            st.write(explain.text_contributions(pkg["text_model"], st.session_state["narr"]["text_masked"], top_class))

# ---------------------------------------------------------------- 5 About
with tab_about:
    st.markdown("""
**Intended use:** research only. To study whether combining CT and clinical narratives helps characterize TB-related and cancer-related findings, and whether the explanations hold up.

**Not for:** diagnosis, triage, screening or any decision about a patient.

**Known limits of this kind of model**
- Labels came from clinical wording, not from histology or microbiology, so the model learns *recorded impressions*.
- The data contain TB and cancer cases only; there are almost no normal scans, so 'Other' is not a normal-scan detector.
- Trained on one hospital's scanners and reporting habits. Performance elsewhere is unknown.
- Heat maps and word lists are aids to inspection, not proof of correct reasoning.
""")
    if card:
        st.json(card)
    else:
        st.info("No model card yet. After training, the notebook writes `model_card.json` with the numbers shown here.")
