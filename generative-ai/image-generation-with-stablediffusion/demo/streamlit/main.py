import streamlit as st
import os
import base64
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path

# Environment configuration
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Generation with StableDifussion",
    page_icon="📷",
    layout="centered"
)

# --- Custom Style ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 0 !important;
        }
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            border: none !important;
        }
        .stTextInput>div>div>input {
            font-size: 16px !important;
            padding: 10px !important;
        }
        .stMarkdown {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0px;
        }
        hr, .stHorizontalRule {
            border-color: rgba(0,77,204,0.20);
        }
        img[alt="HP Logo"],
        img[alt="AI Studio Logo"],
        img[alt="Z by HP Logo"] {
    width: 50px !important;
    height: auto !important;
}

    </style>
""", unsafe_allow_html=True)

# --- Logo ---

def uri_from(path: Path) -> str:
    return f"data:image/{path.suffix[1:].lower()};base64," + base64.b64encode(path.read_bytes()).decode()

assets = Path("assets")
hp_uri = uri_from(assets / "HP-Logo.png")
ais_uri = uri_from(assets / "AI-Studio.png")
zhp_uri = uri_from(assets / "Z-HP-logo.png")

st.markdown(f"""
    <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:1.5rem">
        <img src="{hp_uri}"  alt="HP Logo" style="width:90px;height:auto;">
        <img src="{ais_uri}" alt="AI Studio Logo" style="width:90px;height:auto;">
        <img src="{zhp_uri}" alt="Z by HP Logo" style="width:90px;height:auto;">
    </div>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🖼️ Image Generation with StableDifussion </h1>", unsafe_allow_html=True)

# --- MLflow API Configuration ---
# Standardized MLflow endpoint for containerized deployment
MLFLOW_ENDPOINT = "http://localhost:5002/invocations"
api_url = MLFLOW_ENDPOINT

# --- data inputs ---

prompt = st.text_input("Prompt (image description):")
use_finetuning = st.checkbox("Use fine-tuning")
height = st.number_input("Image height (px):", min_value=1, value=512)
width = st.number_input("Image width (px):", min_value=1, value=512)
num_images = st.number_input("Number of images:", min_value=1, max_value=10, value=1)
num_inference_steps = st.number_input("Number of inference steps:", min_value=1, value=50)


# --- Button to Call the Model ---
if st.button("Get result"):
    if prompt or use_finetuning or height or width or num_images or num_inference_steps not in st.session_state:
        st.warning("⚠️ Please fill all fields!")
    else:
        payload = {
            "inputs": {
                "prompt": [prompt],
                "use_finetuning":[use_finetuning],
                "height":[height],
                "width":[width],
                "num_images": [num_images],
                "num_inference_steps": [num_inference_steps]   
                       },
        }

        with st.spinner("Generating image..."):
            try:
                response = requests.post(api_url, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()

                base64_image = data.get("predictions", [None])[0]

                if base64_image and isinstance(base64_image, str):
                    image_bytes = base64.b64decode(base64_image)
                    image = Image.open(BytesIO(image_bytes))
                    st.session_state["enhanced_image"] = image
                    st.success("✅ Here is your image!")
                else:
                    st.error("❌ No valid image data returned from the model.")
                    st.write("Raw response:", data)

            except requests.exceptions.RequestException as e:
                st.error("❌ Error fetching prediction.")
                st.error(str(e))

# --- Display Enhanced Image and Download Button ---
if "enhanced_image" in st.session_state:
    st.image(st.session_state["enhanced_image"], caption="Output", use_container_width=True)

    buffer = BytesIO()
    st.session_state["enhanced_image"].save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="📥 Download Image",
        data=buffer,
        file_name="image_output.png",
        mime="image/png"
    )

# --- Footer ---
st.markdown(
    """
    > Built with ❤️ using HP AI Studio.
    """,
    unsafe_allow_html=True,
)