import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat, ImageFilter

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(46,204,113,.10), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(0,190,130,.08), transparent 25%),
        linear-gradient(135deg, #08120d 0%, #0a1510 45%, #07100b 100%);
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1250px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: rgba(5, 14, 9, .96);
    border-right: 1px solid rgba(120,255,180,.10);
}

.hero {
    text-align: center;
    padding: 12px 0 24px;
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(83,227,145,.08);
    border: 1px solid rgba(83,227,145,.18);
    color: #77edaa;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .8px;
}

.hero h1 {
    margin: 14px 0 0;
    font-size: clamp(38px, 5vw, 62px);
    line-height: 1.03;
    font-weight: 800;
    letter-spacing: -2.5px;
    color: #f3fff7;
}

.hero h1 span { color: #61e89a; }

.hero p {
    margin: 14px auto 0;
    max-width: 760px;
    color: #a6b6ab;
    font-size: 17px;
    line-height: 1.6;
}

.stat-card {
    padding: 18px 14px;
    border-radius: 18px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.075);
    text-align: center;
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    color: #67e99f;
}

.stat-label {
    margin-top: 4px;
    font-size: 13px;
    color: #94a89b;
}

.upload-card {
    margin: 18px 0 10px;
    padding: 30px 22px;
    border-radius: 24px;
    border: 1px solid rgba(112,255,174,.16);
    background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.02));
    box-shadow: 0 20px 60px rgba(0,0,0,.22);
    text-align: center;
}

.upload-title {
    font-size: 24px;
    font-weight: 700;
    color: #e9fff2;
}

.upload-subtitle {
    margin-top: 6px;
    color: #94aa9d;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.018);
    border: 1px dashed rgba(98,233,154,.35);
    border-radius: 18px;
    padding: 8px;
}

.glass-card {
    padding: 22px;
    border-radius: 22px;
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 15px 45px rgba(0,0,0,.18);
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #ecfff3;
    margin-bottom: 12px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    color: #f5fff8;
    margin: 8px 0;
}

.confidence-value {
    font-size: 34px;
    font-weight: 800;
    color: #67e99f;
}

.pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    background: rgba(80,230,143,.10);
    border: 1px solid rgba(80,230,143,.18);
    color: #78eda9;
}

.info-box {
    padding: 18px;
    border-radius: 17px;
    background: rgba(44,110,80,.14);
    border: 1px solid rgba(106,224,153,.12);
    color: #c5d8cc;
    line-height: 1.6;
}

.step {
    padding: 18px;
    border-radius: 18px;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    height: 100%;
}

.step-number {
    color: #70e99f;
    font-weight: 800;
    font-size: 13px;
}

.step h4 { color: #effff4; margin: 8px 0 6px; }
.step p { color: #93a79a; line-height: 1.55; font-size: 14px; }

.footer {
    margin-top: 44px;
    text-align: center;
    color: #74877a;
    font-size: 13px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,.06);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 38 CLASS NAMES
# =========================================================

class_names = [

    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    possible_models = [

        "models/final_plant_disease_model.keras",

        "models/best_mobilenetv2_finetuned.keras",

        "models/best_mobilenetv2.keras"
    ]

    for path in possible_models:

        try:

            model = tf.keras.models.load_model(path)

            return model, path

        except Exception:

            continue

    return None, None


model, model_path = load_model()


if model is None:

    st.error("❌ No trained model was found.")

    st.info(
        "Please check your models folder."
    )

    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_name(name):

    name = name.replace("___", " - ")
    name = name.replace("_", " ")
    name = name.replace("  ", " ")

    return name


def get_plant_name(name):

    return name.split("___")[0].replace("_", " ")


def get_recommendation(disease):

    disease = disease.lower()

    if "healthy" in disease:

        return (
            "The plant appears healthy. Continue proper watering, "
            "adequate sunlight, good nutrition and regular monitoring."
        )

    if "powdery_mildew" in disease:

        return (
            "Improve air circulation and avoid prolonged moisture "
            "on leaves. Remove severely affected leaves where practical."
        )

    if "rust" in disease:

        return (
            "Remove heavily affected leaves where practical and "
            "improve air circulation. Avoid unnecessary leaf wetness."
        )

    if "blight" in disease:

        return (
            "Remove severely affected plant material, improve airflow "
            "and avoid overhead watering when possible."
        )

    if "bacterial" in disease:

        return (
            "Remove severely affected leaves where practical. Avoid "
            "spreading water between plants and keep foliage dry."
        )

    if "virus" in disease:

        return (
            "Monitor the plant closely and isolate severely affected "
            "plants where practical. Check for possible insect vectors."
        )

    if "spider_mites" in disease:

        return (
            "Inspect the underside of leaves for mites and monitor "
            "the plant regularly."
        )

    if "scab" in disease:

        return (
            "Remove affected leaves where practical and improve "
            "air circulation. Avoid prolonged leaf wetness."
        )

    if "black_rot" in disease:

        return (
            "Remove affected plant material and maintain good plant "
            "sanitation. Avoid prolonged foliage moisture."
        )

    if "leaf_mold" in disease:

        return (
            "Improve ventilation and reduce excessive humidity. "
            "Remove severely affected leaves."
        )

    if "target_spot" in disease:

        return (
            "Improve airflow, remove severely affected leaves and "
            "avoid prolonged moisture on foliage."
        )

    return (
        "Monitor the plant closely and maintain good airflow, "
        "sanitation and appropriate watering."
    )


# =========================================================
# IMAGE QUALITY CHECK
# =========================================================

def check_image_quality(image):

    width, height = image.size

    # Resolution
    if width < 224 or height < 224:

        return False, (
            "The image resolution is too small "
            "for reliable analysis."
        )

    # Brightness
    gray = image.convert("L")

    brightness = ImageStat.Stat(gray).mean[0]

    if brightness < 20:

        return False, (
            "The image is extremely dark."
        )

    if brightness > 250:

        return False, (
            "The image is extremely bright."
        )

    # Basic blur detection
    gray_image = gray.filter(
        ImageFilter.FIND_EDGES
    )

    edge_stat = ImageStat.Stat(
        gray_image
    )

    edge_mean = edge_stat.mean[0]

    if edge_mean < 2:

        return False, (
            "The image appears extremely blurry "
            "or lacks visible details."
        )

    return True, (
        "Image quality looks suitable for analysis."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '''
    <div class="hero">
        <div class="hero-badge">AI-POWERED PLANT SCREENING</div>
        <h1>Know your plant.<br><span>Protect your crop.</span></h1>
        <p>
            Upload a clear leaf image and PlantHealth AI will analyze it
            using a fine-tuned MobileNetV2 model trained on 38 PlantVillage classes.
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown('<div class="stat-card"><div class="stat-value">38</div><div class="stat-label">Supported Classes</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-card"><div class="stat-value">14</div><div class="stat-label">Plant Types</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-card"><div class="stat-value">87.43%</div><div class="stat-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-card"><div class="stat-value">AI</div><div class="stat-label">MobileNetV2</div></div>', unsafe_allow_html=True)

st.write("")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌱 PlantHealth AI")

    st.write(
        "AI-powered plant disease screening system."
    )

    st.divider()

    st.subheader("🤖 AI Model")

    st.write(
        "Architecture: **MobileNetV2**"
    )

    st.write(
        "Training: **Fine-Tuned**"
    )

    st.write(
        "Test Accuracy: **87.43%**"
    )

    st.write(
        "Supported Classes: **38**"
    )

    st.divider()

    st.subheader("🌿 Supported Plants")

    supported_plants = [

        "Apple",
        "Blueberry",
        "Cherry",
        "Corn",
        "Grape",
        "Orange",
        "Peach",
        "Pepper",
        "Potato",
        "Raspberry",
        "Soybean",
        "Squash",
        "Strawberry",
        "Tomato"
    ]

    for plant in supported_plants:

        st.write(
            "• " + plant
        )

    st.divider()

    st.caption(
        "Loaded model:"
    )

    st.caption(
        model_path
    )


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '''
    <div class="upload-card">
        <div class="upload-title">📷 Upload a plant leaf image</div>
        <div class="upload-subtitle">Best results come from a close-up image with one clearly visible leaf.</div>
    </div>
    ''',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(

    "Choose a clear plant leaf image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],

    help=(
        "For best results, upload a close-up image "
        "showing mainly one leaf."
    )
)


# =========================================================
# IF IMAGE UPLOADED
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # =================================================
        # IMAGE PREVIEW + DETAILS
        # =================================================

        col1, col2 = st.columns(
            [1, 1]
        )


        with col1:

            st.subheader(
                "🖼️ Uploaded Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            st.subheader(
                "📋 Image Information"
            )

            width, height = image.size

            st.write(
                f"**Resolution:** "
                f"{width} × {height}"
            )

            st.write(
                f"**Format:** "
                f"{uploaded_file.type}"
            )

            quality_ok, quality_message = (
                check_image_quality(image)
            )


            if quality_ok:

                st.success(
                    "✅ " + quality_message
                )

            else:

                st.warning(
                    "⚠️ " + quality_message
                )


        st.divider()


        # =================================================
        # ANALYZE BUTTON
        # =================================================

        if st.button(

            "🔍 Analyze Plant Health",

            use_container_width=True

        ):

            # ---------------------------------------------
            # IMAGE QUALITY FAILURE
            # ---------------------------------------------

            if not quality_ok:

                st.error(
                    "❌ Image quality is not sufficient."
                )

                st.info(
                    "Please upload a brighter, sharper "
                    "and higher-resolution leaf image."
                )


            # ---------------------------------------------
            # IMAGE QUALITY OK
            # ---------------------------------------------

            else:

                with st.spinner(
                    "🤖 PlantHealth AI is analyzing..."
                ):

                    # Resize
                    img = image.resize(
                        (224, 224)
                    )

                    # Convert to numpy
                    img_array = np.array(
                        img
                    )

                    # Add batch dimension
                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    # Model prediction
                    predictions = model.predict(
                        img_array,
                        verbose=0
                    )[0]


                    # -----------------------------------------
                    # TOP PREDICTION
                    # -----------------------------------------

                    top_index = int(
                        np.argmax(
                            predictions
                        )
                    )

                    predicted_class = (
                        class_names[
                            top_index
                        ]
                    )

                    confidence = float(
                        predictions[
                            top_index
                        ] * 100
                    )


                    # -----------------------------------------
                    # TOP 3
                    # -----------------------------------------

                    top_3 = np.argsort(
                        predictions
                    )[-3:][::-1]


                    second_index = int(
                        top_3[1]
                    )

                    second_confidence = float(
                        predictions[
                            second_index
                        ] * 100
                    )


                    # Difference between #1 and #2
                    margin = (
                        confidence
                        - second_confidence
                    )


                # =================================================
                # PROFESSIONAL RELIABILITY LOGIC
                # =================================================

                RELIABLE_CONFIDENCE = 60.0

                RELIABLE_MARGIN = 10.0


                reliable = (

                    confidence
                    >= RELIABLE_CONFIDENCE

                    and

                    margin
                    >= RELIABLE_MARGIN

                )


                # =================================================
                # ANALYSIS COMPLETE
                # =================================================

                st.success(
                    "✅ Analysis Complete!"
                )


                # =================================================
                # RELIABLE RESULT
                # =================================================

                if reliable:

                    st.markdown(
                        '<div class="result-card">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "🌿 Plant Health Result"
                    )

                    st.markdown(
                        f"## 🦠 "
                        f"{clean_name(predicted_class)}"
                    )

                    st.write(
                        f"### 🎯 Confidence: "
                        f"{confidence:.2f}%"
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


                    # Confidence message
                    st.success(
                        "🟢 Reliable prediction"
                    )


                    # Plant category
                    plant_name = get_plant_name(
                        predicted_class
                    )

                    st.info(
                        f"🌱 Detected plant: "
                        f"**{plant_name}**"
                    )


                    # Recommendation
                    st.subheader(
                        "💡 Recommended Action"
                    )

                    st.info(
                        get_recommendation(
                            predicted_class
                        )
                    )


                # =================================================
                # UNCERTAIN RESULT
                # =================================================

                else:

                    st.markdown(
                        '<div class="warning-card">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "⚠️ No Reliable Diagnosis"
                    )

                    st.write(
                        f"Model confidence: "
                        f"**{confidence:.2f}%**"
                    )

                    st.write(
                        "The AI could not confidently "
                        "match this image to one of "
                        "its supported classes."
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


                    st.warning(
                        "This does NOT necessarily mean "
                        "your image is low quality."
                    )


                    st.info(
                        "🌿 The image may be a real-world "
                        "plant photo that differs from the "
                        "images used to train the model, "
                        "or the plant/disease may not be "
                        "one of the 38 supported classes."
                    )


                    st.subheader(
                        "📷 For Better Results"
                    )

                    st.write(
                        "• Keep mainly one leaf in the frame"
                    )

                    st.write(
                        "• Make the leaf large and clearly visible"
                    )

                    st.write(
                        "• Avoid excessive background"
                    )

                    st.write(
                        "• Make sure disease symptoms are visible"
                    )

                    st.write(
                        "• Use images of supported plants"
                    )


                # =================================================
                # TOP 3 PREDICTIONS
                # =================================================

                st.divider()

                st.subheader(
                    "📊 AI Prediction Analysis"
                )


                if reliable:

                    st.caption(
                        "Top predictions from the AI model."
                    )

                else:

                    st.caption(
                        "These are model candidates only. "
                        "They should NOT be treated as a diagnosis."
                    )


                for rank, index in enumerate(

                    top_3,

                    start=1

                ):

                    score = float(
                        predictions[
                            index
                        ] * 100
                    )

                    name = clean_name(
                        class_names[
                            index
                        ]
                    )

                    st.write(
                        f"**{rank}. "
                        f"{name} — "
                        f"{score:.2f}%**"
                    )

                    st.progress(
                        float(
                            predictions[
                                index
                            ]
                        )
                    )


                # =================================================
                # MODEL DETAILS
                # =================================================

                st.divider()

                with st.expander(
                    "🤖 About PlantHealth AI"
                ):

                    st.write(
                        "PlantHealth AI uses a fine-tuned "
                        "MobileNetV2 image classification model."
                    )

                    st.write(
                        "The model achieved approximately "
                        "**87.43% test accuracy** on 8,179 "
                        "test images."
                    )

                    st.write(
                        "The current system supports "
                        "**38 plant disease/health classes**."
                    )

                    st.write(
                        "Predictions on unsupported or "
                        "significantly different real-world "
                        "images may be uncertain."
                    )


                # =================================================
                # DISCLAIMER
                # =================================================

                st.warning(
                    "⚠️ This application is intended for "
                    "educational and preliminary screening "
                    "purposes. AI predictions should not "
                    "replace professional agricultural diagnosis."
                )


    except Exception as e:

        st.error(
            "❌ Unable to process this image."
        )

        st.code(
            str(e)
        )


# =========================================================
# EMPTY SCREEN
# =========================================================

else:

    st.info(
        "👆 Upload a clear plant leaf image "
        "to begin analysis."
    )


    st.divider()


    st.subheader(
        "✨ How PlantHealth AI Works"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            "### 📷 1. Upload"
        )

        st.write(
            "Upload a clear plant leaf image."
        )


    with col2:

        st.markdown(
            "### 🤖 2. Analyze"
        )

        st.write(
            "The fine-tuned MobileNetV2 model "
            "analyzes the image."
        )


    with col3:

        st.markdown(
            "### 🌿 3. Result"
        )

        st.write(
            "Get a diagnosis only when the "
            "prediction is sufficiently reliable."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🌱 PlantHealth AI | '
    'AI-Based Plant Disease Detection'
    '</div>',
    unsafe_allow_html=True
)