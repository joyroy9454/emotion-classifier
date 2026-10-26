
import streamlit as st
import joblib
import numbers

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Emotion Classifier | NLP Project",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
        color: #182230;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #182230;
    }

    .eyebrow {
        color: #3156a6;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #596579;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 22px;
    }

    .section-note {
        color: #667085;
        font-size: 14px;
        line-height: 1.5;
    }

    /* Readable text input */
    [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #182230 !important;
        -webkit-text-fill-color: #182230 !important;
        caret-color: #182230 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 9px !important;
        font-size: 16px !important;
        line-height: 1.65 !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #7b8494 !important;
        -webkit-text-fill-color: #7b8494 !important;
        opacity: 1 !important;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #3156a6 !important;
        box-shadow: 0 0 0 1px #3156a6 !important;
    }

    /* Buttons */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        border-radius: 8px !important;
        min-height: 44px;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background-color: #3156a6 !important;
        border: 1px solid #3156a6 !important;
        color: #ffffff !important;
    }

    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background-color: #244482 !important;
        border-color: #244482 !important;
    }

    /* Result and input panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-color: #e0e6ef;
        border-radius: 12px;
    }

    [data-testid="stMetric"] {
        background: #f8faff;
        padding: 13px;
        border: 1px solid #e5eaf3;
        border-radius: 9px;
    }

    [data-testid="stMetricLabel"] {
        color: #596579;
    }

    .footer {
        color: #778195;
        font-size: 12px;
        text-align: center;
        padding-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD THE SAVED MODEL FILES
# --------------------------------------------------
@st.cache_resource
def load_model_files():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/count_vectorizer.pkl")
    emotion_mapping = joblib.load("models/emotion_mapping.pkl")

    return model, vectorizer, emotion_mapping


try:
    model, vectorizer, emotion_mapping = load_model_files()
except Exception as error:
    st.error("Unable to load the saved model files.")
    st.write(
        "Check that app.py and these three files are in the "
        "same project folder:"
    )
    st.code(
        "sentiment_model.pkl\n"
        "count_vectorizer.pkl\n"
        "emotion_mapping.pkl"
    )
    st.error(str(error))
    st.stop()


# --------------------------------------------------
# MAP THE PREDICTION TO AN EMOTION NAME
# --------------------------------------------------
def get_emotion_label(prediction):
    if not isinstance(emotion_mapping, dict):
        return str(prediction)

    prediction_text = str(prediction)

    # Mapping format: {0: "sadness", 1: "joy"}
    if prediction in emotion_mapping:
        value = emotion_mapping[prediction]

        if not isinstance(value, numbers.Number):
            return str(value)

    if prediction_text in emotion_mapping:
        value = emotion_mapping[prediction_text]

        if not isinstance(value, numbers.Number):
            return str(value)

    # Mapping format: {"sadness": 0, "joy": 1}
    for emotion_name, class_value in emotion_mapping.items():
        if str(class_value) == prediction_text:
            return str(emotion_name)

    return f"Class {prediction_text}"


# --------------------------------------------------
# SESSION STATE
# Keeps the last prediction visible until a new one is made
# --------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
st.markdown(
    '<div class="eyebrow">MACHINE LEARNING · NATURAL LANGUAGE PROCESSING</div>',
    unsafe_allow_html=True,
)

st.title("Emotion Classifier")

st.markdown(
    """
    <div class="subtitle">
        Analyse the emotion expressed in a sentence using a trained
        machine learning model. Enter your text to get a prediction
        and explore the model's estimated probabilities.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
input_column, result_column = st.columns(
    [1, 1],
    gap="large",
)

# --------------------------------------------------
# INPUT PANEL
# --------------------------------------------------
with input_column:
    with st.container(border=True):
        st.subheader("Analyse your text")

        st.markdown(
            '<div class="section-note">'
            'Write a sentence or paste a short paragraph below.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        with st.form("emotion_form"):
            user_text = st.text_area(
                "Text to analyse",
                placeholder=(
                    "Example: I feel happy because I had "
                    "a wonderful day."
                ),
                height=210,
                max_chars=5000,
            )

            word_count = len(user_text.split())

            st.caption(
                f"{word_count} words · {len(user_text)}/5000 characters"
            )

            submitted = st.form_submit_button(
                "Predict emotion",
                type="primary",
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-note">'
            'The text is transformed using the vectorizer saved '
            'from model training.'
            '</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# RUN THE PREDICTION
# --------------------------------------------------
if submitted:
    if not user_text.strip():
        st.session_state.prediction_result = None
        st.session_state.prediction_error = (
            "Please enter some text before predicting."
        )

    else:
        try:
            text_vector = vectorizer.transform([user_text.strip()])

            prediction = model.predict(text_vector)[0]
            emotion = get_emotion_label(prediction)

            probability_rows = []
            top_probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(text_vector)[0]

                probability_rows = sorted(
                    [
                        (
                            get_emotion_label(class_value),
                            float(probability),
                        )
                        for class_value, probability in zip(
                            model.classes_,
                            probabilities,
                        )
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )

                top_probability = max(
                    float(probability)
                    for probability in probabilities
                )

            st.session_state.prediction_result = {
                "emotion": emotion,
                "top_probability": top_probability,
                "probabilities": probability_rows,
                "word_count": word_count,
            }

            st.session_state.prediction_error = None

        except Exception as error:
            st.session_state.prediction_result = None
            st.session_state.prediction_error = str(error)


# --------------------------------------------------
# RESULT PANEL
# --------------------------------------------------
with result_column:
    with st.container(border=True):
        st.subheader("Prediction result")

        st.markdown(
            '<div class="section-note">'
            'Your model output and probability breakdown appear here.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        if st.session_state.prediction_error:
            st.warning(st.session_state.prediction_error)

        elif st.session_state.prediction_result is None:
            st.markdown("### Ready to analyse")

            st.write(
                "Your result will appear here after you enter text "
                "and select **Predict emotion**."
            )

            st.divider()

            st.caption(
                "Tip: Try a complete sentence to give the model "
                "more context."
            )

        else:
            result = st.session_state.prediction_result
            emotion = result["emotion"]

            st.caption("PREDICTED EMOTION")

            st.markdown(
                f"<h2 style='color:#244482; margin-top:0;'>"
                f"{emotion.title()}</h2>",
                unsafe_allow_html=True,
            )

            if emotion.startswith("Class "):
                st.warning(
                    "A class number was returned instead of an emotion "
                    "name. Check the label mapping saved from your notebook."
                )

            if result["top_probability"] is not None:
                st.metric(
                    "Top-class probability",
                    f"{result['top_probability'] * 100:.1f}%",
                )

                st.caption(
                    "This is the model's estimated probability, "
                    "not a guarantee that the prediction is correct."
                )

            if result["probabilities"]:
                st.divider()
                st.markdown("#### Probability breakdown")

                for label, probability in result["probabilities"]:
                    label_column, score_column = st.columns(
                        [3, 1]
                    )

                    with label_column:
                        st.write(label.title())

                    with score_column:
                        st.write(f"{probability * 100:.1f}%")

                    st.progress(probability)


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------
st.write("")
st.divider()

st.subheader("About this project")

info_model, info_features, info_classes = st.columns(3)

with info_model:
    st.markdown("**Model**")
    st.write(model.__class__.__name__)

with info_features:
    st.markdown("**Text representation**")
    st.write(vectorizer.__class__.__name__)

with info_classes:
    st.markdown("**Output classes**")
    st.write(
        str(len(model.classes_))
        if hasattr(model, "classes_")
        else "Available"
    )

st.markdown(
    """
    <div class="footer">
        Built with Python, Streamlit and Scikit-learn.
        Model predictions are estimates and may not capture every nuance
        of human language.
    </div>
    """,
    unsafe_allow_html=True,
)
