import streamlit as st
from pathlib import Path

from pneumonia.model_loader import load_finetuned_model
from pneumonia.input_validation import is_valid_extension, is_valid_image
from pneumonia.predict import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "best_model.pth"


@st.cache_resource
def get_model():
    """Loads the model once and keeps it in memory across reruns."""
    return load_finetuned_model(CHECKPOINT_PATH)


st.title("Pneumonia Detection")
st.write("Upload a chest X-ray to screen for pneumonia.")

uploaded = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    if not is_valid_extension(uploaded.name) or not is_valid_image(uploaded):
        st.error("Invalid file. Please upload a readable JPG or PNG image.")
    else:
        st.image(uploaded, caption="Uploaded X-ray", use_container_width=True)
        uploaded.seek(0)

        model = get_model()
        results = predict(model, uploaded)

        st.metric("Prediction", results["class_name"])
        st.metric("Confidence", f"{results['confidence']:.2%}")

st.warning(
    "This is a demonstration project, not a validated medical device. "
    "On the test set it missed 1 pneumonia case out of 390, but flagged "
    "34% of healthy patients as positive. Every result must be reviewed "
    "by a qualified radiologist."
)