"""Brain Tumor MRI Classifier - Streamlit app with user accounts, saved prediction history and Grad-CAM.
Run:  streamlit run app.py     (from the project folder; needs models/*.h5 and models/class_names.json)"""
import glob
import hashlib
import io
import json
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

import src.models  # noqa: F401  (registers the custom augmentation layer needed to load the .h5 files)
from src import auth_db as db
from src.inference import grad_cam, overlay, predict, preprocess

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")
db.init_db()

CONF_THRESHOLD = 0.60
MODELS_DIR, RESULTS_DIR = "models", "results"
DISCLAIMER = "Educational decision-support demo. NOT a medical device and not for diagnosis. Do not upload real patient data."
ss = st.session_state
ss.setdefault("user", None)
ss.setdefault("results", {})     # (image hash, model) -> cached prediction
ss.setdefault("saved", {})       # (image hash, model) -> id of the saved history record


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model... the first load can take a minute for large models")
def load_model(path):
    model = keras.models.load_model(path, compile=False)
    # warm-up: the very first call is the slowest (graph set-up), so do it once here and not when the user clicks Classify
    model(np.zeros((1, model.input_shape[1], model.input_shape[2], 3), dtype="float32"), training=False)
    return model


@st.cache_resource(show_spinner="Downloading model files (first start only, can take a minute)...")
def ensure_remote_models():
    """Optional: fetch model files that are too big for the GitHub repo (limit 100 MB per file).
    List them in models/model_urls.json as {"best_model_slim.h5": "https://github.com/<user>/<repo>/releases/download/<tag>/best_model_slim.h5"}."""
    cfg, errors = os.path.join(MODELS_DIR, "model_urls.json"), []
    if not os.path.exists(cfg):
        return errors
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in json.load(open(cfg)).items():
        dest = os.path.join(MODELS_DIR, os.path.basename(name))
        if os.path.exists(dest):
            continue
        if not (url.startswith("https://") or url.startswith("http://localhost")):
            errors.append(f"{name}: only https:// links are allowed")
            continue
        try:
            with urllib.request.urlopen(url, timeout=180) as resp, open(dest + ".part", "wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(dest + ".part", dest)                 # a half-downloaded file is never used
        except Exception as e:
            errors.append(f"{name}: download failed ({e})")
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")
    return errors


def model_files():
    return sorted(glob.glob(f"{MODELS_DIR}/*.h5") + glob.glob(f"{MODELS_DIR}/*.keras"))


def default_model_index(files):
    for prefix in ("best_model_slim", "best_model"):
        for i, f in enumerate(files):
            if os.path.basename(f).startswith(prefix):
                return i
    return 0


def class_names():
    return json.load(open(f"{MODELS_DIR}/class_names.json"))


# --------------------------------------------------------------------- pages
def page_auth():
    st.title("🧠 Brain Tumor MRI Classifier")
    st.caption(DISCLAIMER)
    tab_login, tab_register = st.tabs(["Log in", "Register"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                user, msg = db.authenticate(username, password)
                if user:
                    ss.user = user
                    st.rerun()
                st.error(msg)
    with tab_register:
        with st.form("register_form"):
            full_name = st.text_input("Full name (optional)")
            new_user = st.text_input("Choose a username")
            pw1 = st.text_input("Password (8+ characters, letters and numbers)", type="password")
            pw2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create account"):
                if pw1 != pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.register_user(new_user, pw1, full_name)
                    (st.success if ok else st.error)(msg)


def make_cam(model, res, top):
    """Grad-CAM overlay for a cached prediction (slower than the prediction itself, so it is optional)."""
    cam, _, _ = grad_cam(model, res["x"], top)
    return overlay(res["x"], cam)


def page_predict():
    st.header("Predict tumor type")
    for err in ensure_remote_models():
        st.warning(err)
    files = model_files()
    if not files or not os.path.exists(f"{MODELS_DIR}/class_names.json"):
        st.error("No trained model found. Run `python -m src.train --data_dir <dataset>` first.")
        return
    labels = {f: f"{os.path.basename(f)} ({os.path.getsize(f) / 1e6:.0f} MB)" for f in files}
    choice = st.sidebar.selectbox("Model", files, index=default_model_index(files), format_func=labels.get)
    st.sidebar.caption("MobileNetV2 is the fastest; ResNet50V2 is the most accurate but loads slowly the first time.")
    auto_cam = st.sidebar.checkbox("Compute Grad-CAM with every classification (slower)", value=False)
    classes = class_names()
    st.sidebar.write(f"Classes: {', '.join(classes)}")

    uploads = st.file_uploader("Upload MRI image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    go = st.button("🔍 Classify", type="primary", disabled=not uploads)

    # ---- work happens ONLY when the button is pressed; results are cached, so later clicks (Save, Grad-CAM) are instant ----
    if go:
        with st.spinner("Loading model and classifying..."):
            model = load_model(choice)
            size = model.input_shape[1]
            for f in uploads:
                data = f.getvalue()
                key = (hashlib.md5(data).hexdigest(), choice)
                if key in ss.results:
                    continue
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    st.error(f"{f.name}: not a readable image.")
                    continue
                x = preprocess(img, size)
                res = {"img": img, "x": x, "probs": predict(model, x)}
                if auto_cam:
                    res["cam_img"] = make_cam(model, res, int(np.argmax(res["probs"])))
                ss.results[key] = res

    # ---- show the cached results of the uploaded files ----
    for f in uploads or []:
        key = (hashlib.md5(f.getvalue()).hexdigest(), choice)
        res = ss.results.get(key)
        if res is None:
            st.info(f"**{f.name}**: press **Classify** to get the result.")
            continue
        probs, top = res["probs"], int(np.argmax(res["probs"]))

        st.subheader(f.name)
        c1, c2 = st.columns(2)
        c1.image(res["img"], caption="Uploaded image")
        if "cam_img" in res:
            c2.image(res["cam_img"], caption="Grad-CAM: regions that influenced the prediction")
        elif c2.button("🔥 Show Grad-CAM heat map", key=f"cam_{key[0]}_{key[1]}"):
            with st.spinner("Computing Grad-CAM..."):
                res["cam_img"] = make_cam(load_model(choice), res, top)
            c2.image(res["cam_img"], caption="Grad-CAM: regions that influenced the prediction")
        m1, m2 = st.columns(2)
        m1.metric("Predicted tumor type", classes[top], f"{probs[top]:.1%} confidence")
        m2.bar_chart(pd.DataFrame({"probability": probs}, index=classes))
        if probs[top] < CONF_THRESHOLD:
            st.warning("Low confidence - this scan should be reviewed by a specialist.")

        if key in ss.saved:
            st.success(f"Saved to your history (record #{ss.saved[key]}).")
        else:
            note = st.text_input("Optional note", key=f"note_{key[0]}_{key[1]}", max_chars=300)
            if st.button("💾 Save to my history", key=f"save_{key[0]}_{key[1]}"):
                ss.saved[key] = db.save_prediction(ss.user["id"], f.name, os.path.basename(choice), classes[top],
                                                   probs[top], dict(zip(classes, probs)), res["img"], note)
                st.rerun()
        st.divider()


def page_history():
    st.header("My prediction history")
    rows = db.list_predictions(ss.user["id"])
    if not rows:
        st.info("Nothing saved yet. Make a prediction and click *Save to my history*.")
        return
    df = pd.DataFrame([{"id": r["id"], "date": r["created_at"], "file": r["filename"], "model": r["model_name"],
                        "prediction": r["predicted_class"], "confidence": round(r["confidence"], 3), "note": r["note"]} for r in rows])
    a, b, c = st.columns(3)
    a.metric("Saved predictions", len(df))
    b.metric("Most common result", df["prediction"].mode().iat[0])
    c.metric("Average confidence", f"{df['confidence'].mean():.1%}")
    chosen = st.multiselect("Filter by prediction", sorted(df["prediction"].unique()))
    view = df[df["prediction"].isin(chosen)] if chosen else df
    st.dataframe(view, hide_index=True)
    st.download_button("⬇️ Download as CSV", view.to_csv(index=False).encode(), "my_predictions.csv", "text/csv")
    st.bar_chart(df["prediction"].value_counts())
    st.subheader("Details")
    for r in [r for r in rows if r["id"] in set(view["id"])][:50]:
        with st.expander(f"#{r['id']} · {r['predicted_class']} ({r['confidence']:.0%}) · {r['created_at']}"):
            c1, c2 = st.columns(2)
            if r["image_path"] and os.path.exists(r["image_path"]):
                c1.image(r["image_path"], caption=r["filename"])
            c2.bar_chart(pd.DataFrame({"probability": r["probs"]}))
            st.caption(f"Model: {r['model_name']}" + (f" · Note: {r['note']}" if r["note"] else ""))
            if st.button("🗑️ Delete this record", key=f"del_{r['id']}"):
                db.delete_prediction(ss.user["id"], r["id"])
                st.rerun()


def page_results():
    st.header("Model results")
    csv = f"{RESULTS_DIR}/model_comparison.csv"
    if not os.path.exists(csv):
        st.info("No results yet. Run training (`python -m src.train ...`) and copy the `results/` folder next to this app.")
        return
    st.subheader("Model comparison (test set)")
    st.dataframe(pd.read_csv(csv), hide_index=True)
    if os.path.exists(f"{RESULTS_DIR}/model_comparison.png"):
        st.image(f"{RESULTS_DIR}/model_comparison.png")
    names = sorted(p.split("/")[-1].replace("_classification_report.txt", "") for p in glob.glob(f"{RESULTS_DIR}/*_classification_report.txt"))
    if names:
        name = st.selectbox("Details for model", names)
        c1, c2 = st.columns(2)
        if os.path.exists(f"{RESULTS_DIR}/{name}_confusion_matrix.png"):
            c1.image(f"{RESULTS_DIR}/{name}_confusion_matrix.png")
        if os.path.exists(f"{RESULTS_DIR}/{name}_history.png"):
            c2.image(f"{RESULTS_DIR}/{name}_history.png")
        st.code(open(f"{RESULTS_DIR}/{name}_classification_report.txt").read())
    if os.path.exists(f"{RESULTS_DIR}/misclassified.png"):
        st.subheader("Most confident mistakes")
        st.image(f"{RESULTS_DIR}/misclassified.png")
    else:
        st.caption("Tip: run `python -m src.analyze_errors --data_dir <dataset>` to add an error-analysis picture here.")


def page_about():
    st.header("About & limitations")
    st.markdown("""
**What it does:** classifies a brain MRI image into glioma, meningioma, pituitary tumor or no tumor using deep-learning models
(custom CNN and transfer learning), and shows a Grad-CAM heat map of the image regions that influenced the prediction.

**Limitations**
- Small public dataset (about 2,400 images, a few hundred test images): results are approximate and not externally validated.
- Meningioma is the hardest class and is often confused with pituitary tumors.
- Grad-CAM is a visual aid, not proof that the model reasons like a radiologist.
- Educational demo, **not a medical device**; never use it for diagnosis or treatment decisions.

**Privacy:** accounts and saved predictions are stored in a local SQLite database (`data/app.db`) and saved images in `data/uploads/`.
Passwords are stored only as salted hashes. Do not upload real patient data. You can delete single records or your whole account at any time.
""")


# ---------------------------------------------------------------------- main
if ss.user is None:
    page_auth()
    st.stop()

with st.sidebar:
    st.write(f"👤 **{ss.user['full_name']}**")
    if st.button("Log out"):
        ss.user = None
        ss.results, ss.saved = {}, {}
        st.rerun()
    page = st.radio("Go to", ["Predict", "My history", "Model results", "About"])
    with st.expander("Account settings"):
        if st.checkbox("I want to permanently delete my account and all saved predictions"):
            if st.button("Delete my account"):
                db.delete_user(ss.user["id"])
                ss.user = None
                st.rerun()
st.caption(DISCLAIMER)
{"Predict": page_predict, "My history": page_history, "Model results": page_results, "About": page_about}[page]()
