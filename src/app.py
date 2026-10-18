
import streamlit as st
from PIL import Image

from facenet_model import load_models, compare_faces


# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="FaceNet Face Verification",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 FaceNet Face Verification")

st.write(
    "Upload two face images and compare their facial embeddings "
    "using a pretrained FaceNet model."
)

st.info(
    "Use images you have permission to process. "
    "This demo is not proof of identity and is not designed "
    "for security-critical decisions."
)


# -------------------------
# Model loading
# -------------------------

@st.cache_resource
def get_models():
    """Load the models once and reuse them across reruns."""
    return load_models()


# -------------------------
# Image upload interface
# -------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("First image")

    file1 = st.file_uploader(
        "Upload the reference face",
        type=["jpg", "jpeg", "png"],
        key="first_image",
    )

    if file1 is not None:
        image1 = Image.open(file1).convert("RGB")
        st.image(
            image1,
            caption="Reference image",
            width="stretch",
        )

with col2:
    st.subheader("Second image")

    file2 = st.file_uploader(
        "Upload the face to compare",
        type=["jpg", "jpeg", "png"],
        key="second_image",
    )

    if file2 is not None:
        image2 = Image.open(file2).convert("RGB")
        st.image(
            image2,
            caption="Comparison image",
            width="stretch",
        )


# -------------------------
# Face verification
# -------------------------

THRESHOLD = 0.70

if st.button(
    "Verify faces",
    type="primary",
    width="stretch",
):
    if file1 is None or file2 is None:
        st.warning("Please upload both images before continuing.")

    else:
        try:
            with st.spinner(
                "Loading FaceNet and comparing faces..."
            ):
                # Reopen uploads as RGB images.
                image1 = Image.open(file1).convert("RGB")
                image2 = Image.open(file2).convert("RGB")

                detector, model = get_models()

                score = compare_faces(
                    image1,
                    image2,
                    detector,
                    model,
                )

            st.divider()
            st.subheader("Comparison results")

            st.metric(
                "Cosine similarity",
                f"{score:.4f}",
            )

            if score >= THRESHOLD:
                st.success(
                    "Match under the current demo threshold"
                )
            else:
                st.warning(
                    "Non-match under the current demo threshold"
                )

            st.caption(
                f"Demo threshold: {THRESHOLD:.2f}. "
                "This is an illustrative cutoff, not a validated "
                "accuracy guarantee. Calibrate it on a suitable "
                "evaluation dataset before drawing conclusions."
            )

        except ValueError as error:
            st.error(str(error))

        except Exception:
            st.error(
                "An unexpected error occurred. Check the "
                "VS Code terminal for details."
            )
            st.exception(Exception("See the application traceback above."))
