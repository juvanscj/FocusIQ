import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# FOCUSIQ | STUDENT PRODUCTIVITY PREDICTION
# Exactly five inputs; no dataset is loaded or displayed.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "focusiqfinal.pkl"
SCALER_PATH = BASE_DIR / "focusiq_scalerfinal.pkl"

FEATURES = [
    "study_hours_per_day",
    "sleep_hours",
    "stress_level",
    "focus_score",
    "total_screen_time_hours"
]

st.set_page_config(
    page_title="FOCUSIQ | Student Focus Lab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- THEME --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(32,104,255,.15), transparent 35%),
      linear-gradient(145deg,#070b14 0%,#0a1020 55%,#080d18 100%);
    color:#eef4ff;
    font-family:'DM Sans',sans-serif;
}

[data-testid="stHeader"] {
    background:transparent;
}

.block-container {
    max-width:1500px;
    padding-top:2.2rem;
    padding-bottom:3rem;
}

h1,h2,h3,h4 {
    color:#f4f7ff !important;
    font-family:'Space Grotesk',sans-serif !important;
}

p,label {
    color:#c7d3e8;
}

.eyebrow {
    color:#68a5ff;
    font-size:.78rem;
    letter-spacing:.22em;
    font-weight:700;
}

.hero-title {
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2.7rem,6vw,4.5rem);
    line-height:1.04;
    font-weight:700;
    letter-spacing:-.055em;
    color:#f5f8ff;
    margin-top:1rem;
}

.hero-title span {
    color:#5c9dff;
}

.hero-copy {
    max-width:720px;
    color:#a9bad4;
    font-size:1.05rem;
    line-height:1.75;
    margin-top:1rem;
}

.blue-line {
    height:3px;
    width:54px;
    background:#438dff;
    border-radius:8px;
    margin:1.5rem 0;
}

.blue-line.center {
    margin:1.5rem auto;
}

div[data-testid="stForm"] {
    background:rgba(13,22,39,.82);
    border:1px solid rgba(115,157,220,.2);
    border-radius:20px;
    padding:1.4rem;
}

.input-heading {
    font-family:'Space Grotesk',sans-serif;
    color:#dce8ff;
    font-size:1.35rem;
    font-weight:700;
    margin-bottom:1.2rem;
}

/* ---------- SIDE RATING GUIDE CARDS ---------- */

.help-card {
    background:linear-gradient(145deg,#111e34,#0b1425);
    border:1px solid rgba(80,145,255,.24);
    border-radius:16px;
    padding:20px 20px 12px 20px;
    box-sizing:border-box;
}

.help-card.left {
    border-left:3px solid #438dff;
}

.help-card.right {
    border-right:3px solid #438dff;
}

.help-card h4 {
    color:#76adff !important;
    margin:0 0 8px 0;
    font-size:1.05rem;
}

.help-card p.q {
    color:#bdcbe0;
    font-size:.86rem;
    line-height:1.55;
    margin:0 0 12px 0;
}

.guide-row {
    padding:10px 0;
    border-top:1px solid rgba(115,157,220,.14);
}

.guide-head {
    display:flex;
    gap:10px;
    align-items:baseline;
}

.guide-score {
    color:#76adff;
    font-weight:700;
    font-size:.88rem;
    min-width:52px;
}

.guide-meaning {
    color:#f1f6ff;
    font-weight:600;
    font-size:.9rem;
}

.guide-exp {
    color:#a9bad4;
    font-size:.8rem;
    line-height:1.45;
    margin-top:3px;
    padding-left:62px;
}

.tip-card {
    background:#101d32;
    border:1px solid rgba(80,145,255,.2);
    border-left:3px solid #438dff;
    border-radius:12px;
    padding:13px 16px;
    margin:9px 0;
    color:#d6e2f5;
    line-height:1.55;
    text-align:left;
}

.result-panel {
    background:linear-gradient(145deg,#111e34,#0b1425);
    border:1px solid rgba(80,145,255,.25);
    border-radius:17px;
    padding:22px;
    color:#dce8ff;
    line-height:1.65;
    text-align:center;
}

/* ---------- CENTERED PREDICTION SECTION ---------- */

.center-title {
    text-align:center;
    font-family:'Space Grotesk',sans-serif;
    color:#f4f7ff;
    font-size:1.75rem;
    font-weight:700;
    margin:1.4rem 0 .8rem 0;
}

.center-sub {
    text-align:center;
    font-family:'Space Grotesk',sans-serif;
    color:#f4f7ff;
    font-size:1.4rem;
    font-weight:700;
    margin:1.6rem 0 .8rem 0;
}

div[data-testid="stAlert"] {
    text-align:center;
}

div[data-testid="stAlert"] p {
    text-align:center;
    width:100%;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
div.stDownloadButton > button {
    background:linear-gradient(100deg,#2877f5,#4b9aff);
    color:white;
    border:1px solid #5597ff;
    border-radius:11px;
    min-height:3rem;
    font-weight:700;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
div.stDownloadButton > button:hover {
    color:white;
    border-color:#a6caff;
    filter:brightness(1.08);
}

[data-testid="stMetric"] {
    background:rgba(20,34,58,.75);
    border:1px solid rgba(115,157,220,.16);
    padding:14px;
    border-radius:13px;
    text-align:center;
}

[data-testid="stMetricLabel"] {
    color:#9eafca !important;
    justify-content:center;
}

[data-testid="stMetricValue"] {
    color:#f5f8ff !important;
    justify-content:center;
}

hr {
    border-color:rgba(115,157,220,.17);
}

.footer {
    color:#7185a4;
    font-size:.8rem;
    text-align:center;
    padding:22px 0 5px;
}
/* ---------- VISUAL ENHANCEMENTS ---------- */
.stApp {
    background:
      radial-gradient(ellipse at 8% 8%, rgba(48,116,255,.22), transparent 32%),
      radial-gradient(ellipse at 92% 18%, rgba(91,75,220,.13), transparent 28%),
      linear-gradient(145deg,#060b16 0%,#0a1426 52%,#080d19 100%);
}
.block-container {
    max-width: 1540px;
    padding-left: clamp(1rem, 4vw, 3.5rem);
    padding-right: clamp(1rem, 4vw, 3.5rem);
}
.hero-visual {
    border: 1px solid rgba(127,174,255,.28);
    border-radius: 24px;
    padding: 9px;
    overflow: hidden;
    background: linear-gradient(145deg,rgba(68,130,255,.20),rgba(15,28,50,.72));
    box-shadow: 0 22px 60px rgba(0,0,0,.28);
}
.hero-visual img {
    display: block;
    width: 100%;
    height: 300px;
    border-radius: 17px;
    object-fit: cover;
}
div[data-testid="stForm"] {
    background: linear-gradient(145deg,rgba(18,31,53,.94),rgba(9,17,31,.96));
    border: 1px solid rgba(115,157,220,.27);
    box-shadow: 0 18px 50px rgba(0,0,0,.20);
    backdrop-filter: blur(12px);
}
.help-card, .result-panel {
    box-shadow: 0 14px 35px rgba(0,0,0,.18);
    backdrop-filter: blur(10px);
}
div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: .25rem;
}
div[data-testid="stSlider"] [role="slider"] {
    box-shadow: 0 0 0 5px rgba(67,141,255,.13);
}
div[data-testid="stFormSubmitButton"] > button,
div.stButton > button,
div.stDownloadButton > button {
    box-shadow: 0 8px 22px rgba(40,119,245,.22);
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
div[data-testid="stFormSubmitButton"] > button:hover,
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(40,119,245,.32);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg,rgba(24,43,72,.9),rgba(12,23,41,.9));
    border: 1px solid rgba(115,157,220,.22);
    box-shadow: 0 10px 25px rgba(0,0,0,.14);
}
div[data-testid="stMarkdownContainer"] a {
    color: #83b5ff;
}
@media (max-width: 900px) {
    .hero-title { font-size: clamp(2.4rem, 10vw, 3.7rem); }
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stForm"] { padding: 1rem; }
}

</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL AND SCALER --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH.name}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {SCALER_PATH.name}"
        )

    return joblib.load(SCALER_PATH)


try:
    model = load_model()
    scaler = load_scaler()

except Exception as exc:

    st.error("Unable to load the trained model or scaler.")

    st.code(str(exc))

    st.info(
        "Place focusiqfinal.pkl and focusiq_scalerfinal.pkl "
        "in the same folder as app.py."
    )

    st.stop()


# -------------------- GUIDE HELPER --------------------

def guide_card(side, title, question, rows):

    html = (
        f'<div class="help-card {side}">'
        f'<h4>{title}</h4>'
        f'<p class="q">{question}</p>'
    )

    for score, meaning, explanation in rows:

        html += (
            '<div class="guide-row">'
            '<div class="guide-head">'
            f'<span class="guide-score">{score}</span>'
            f'<span class="guide-meaning">{meaning}</span>'
            '</div>'
            f'<div class="guide-exp">{explanation}</div>'
            '</div>'
        )

    html += '</div>'

    return html


FOCUS_ROWS = [
    (
        "0–30",
        "Low Focus",
        "Easily distracted and finds it difficult to concentrate."
    ),
    (
        "31–60",
        "Moderate Focus",
        "Can concentrate but loses attention sometimes."
    ),
    (
        "61–80",
        "Good Focus",
        "Usually stays attentive while studying."
    ),
    (
        "81–100",
        "Excellent Focus",
        "Maintains strong concentration with fewer distractions."
    ),
]

STRESS_ROWS = [
    ("0–2", "Very Low", "Feeling relaxed and comfortable."),
    ("3–4", "Low", "Experiencing very little pressure."),
    ("5–6", "Moderate", "Some pressure due to studies or deadlines."),
    ("7–8", "High", "Feeling overwhelmed by academic responsibilities."),
    ("9–10", "Very High", "Experiencing significant pressure."),
]


# -------------------- HERO --------------------

hero_text, hero_image = st.columns([1.35, 0.85], gap="large", vertical_alignment="center")

with hero_text:
    st.markdown(
        '<div class="eyebrow">FOCUSIQ / STUDENT FOCUS LAB</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">Make your focus<br>'
        '<span>work for you.</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-copy">A quick snapshot of your study routine. '
        'Add your daily habits, get a productivity estimate, and leave '
        'with a few practical changes to try.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="blue-line"></div>',
        unsafe_allow_html=True
    )

with hero_image:
    st.markdown(
        '''
        <div class="hero-visual">
            <img
                src="https://images.unsplash.com/photo-1434030216411-0b793f4b4173?auto=format&fit=crop&w=1100&q=85"
                alt="Student writing notes at a study desk"
            />
        </div>
        ''',
        unsafe_allow_html=True
    )
    st.caption("A calmer space. A clearer mind. Better focus.")


st.markdown("### Build your daily snapshot")

st.caption(
    "Enter your usual daily values. The prediction uses exactly "
    "the five features used to train your machine learning model."
)


# -------------------- THREE COLUMNS --------------------

left_side, main, right_side = st.columns(
    [1.15, 2.3, 1.15],
    gap="large"
)

with left_side:

    st.markdown(
        guide_card(
            "left",
            "🎯 Focus Score — Rating Guide",
            "How well can you concentrate during your study sessions?",
            FOCUS_ROWS
        ),
        unsafe_allow_html=True
    )


with right_side:

    st.markdown(
        guide_card(
            "right",
            "⚡ Stress Level — Rating Guide",
            "How much academic or personal pressure do you usually experience?",
            STRESS_ROWS
        ),
        unsafe_allow_html=True
    )


with main:

    # ---------------- FIVE INPUTS ONLY ----------------

    with st.form("student_productivity_form"):

        left, right = st.columns(2, gap="large")

        with left:

            st.markdown(
                '<div class="input-heading">01 — Study rhythm</div>',
                unsafe_allow_html=True
            )

            study_hours = st.slider(
                "Study hours",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help="Total hours spent studying or learning in a usual day."
            )

            focus_score = st.slider(
                "Focus score (0–100)",
                min_value=0,
                max_value=100,
                value=70,
                step=1,
                help="0 = unable to concentrate; 100 = excellent concentration."
            )

        with right:

            st.markdown(
                '<div class="input-heading">02 — Energy & environment</div>',
                unsafe_allow_html=True
            )

            sleep_hours = st.slider(
                "Sleep hours",
                min_value=0.0,
                max_value=12.0,
                value=7.0,
                step=0.5,
                help="Total hours of sleep in a usual day."
            )

            screen_time = st.slider(
                "Total screen time (hours)",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                help="Combined daily phone, YouTube, gaming and social-media time."
            )

            stress_level = st.slider(
                "Stress level (0–10)",
                min_value=0,
                max_value=10,
                value=5,
                step=1,
                help="0 = no stress; 10 = very high stress."
            )

        st.markdown(
            '<p style="color:#91a4c3;font-size:.82rem;">'
            'Note: Screen time means combined daily hours spent on phone, '
            'YouTube, gaming and social media. Focus (0–100) and stress (0–10) '
            'are self-ratings.'
            '</p>',
            unsafe_allow_html=True
        )

        submitted = st.form_submit_button(
            "✦  PREDICT MY PRODUCTIVITY",
            use_container_width=True
        )


    # ---------------- PREDICTION ----------------

    if submitted:

        input_data = pd.DataFrame(
            [[
                study_hours,
                sleep_hours,
                stress_level,
                focus_score,
                screen_time
            ]],
            columns=FEATURES
        )

        try:

            # Validate scaler feature names.

            if hasattr(scaler, "feature_names_in_"):

                scaler_columns = list(scaler.feature_names_in_)

                if set(scaler_columns) != set(FEATURES):

                    raise ValueError(
                        "Scaler feature names do not match the five app inputs. "
                        f"Scaler expects: {scaler_columns}"
                    )

                input_for_scaler = input_data[scaler_columns]

            else:

                input_for_scaler = input_data


            # Validate scaler input count.

            if hasattr(scaler, "n_features_in_"):

                if scaler.n_features_in_ != len(FEATURES):

                    raise ValueError(
                        f"Scaler expects {scaler.n_features_in_} features, "
                        f"but the app supplies {len(FEATURES)}."
                    )


            # Scale the input.

            scaled_input = scaler.transform(input_for_scaler)


            # Validate model input count.

            if hasattr(model, "n_features_in_"):

                if model.n_features_in_ != len(FEATURES):

                    raise ValueError(
                        f"Model expects {model.n_features_in_} features, "
                        f"but the app supplies {len(FEATURES)}."
                    )


            # Predict productivity.

            prediction = model.predict(scaled_input)[0]

            prediction_label = str(prediction).strip().title()


            # ---------------- DISPLAY RESULT ----------------

            st.markdown(
                '<div class="blue-line center"></div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="center-title">'
                'Your productivity result</div>',
                unsafe_allow_html=True
            )


            if prediction_label.lower() == "high":

                st.success("🟢 HIGH PRODUCTIVITY")

                result_message = (
                    "Your predicted productivity category is High. "
                    "Keep maintaining your study routine and healthy habits."
                )


            elif prediction_label.lower() == "medium":

                st.warning("🟡 MEDIUM PRODUCTIVITY")

                result_message = (
                    "Your predicted productivity category is Medium. "
                    "Consistent study sessions and fewer distractions may help."
                )


            elif prediction_label.lower() == "low":

                st.error("🔴 LOW PRODUCTIVITY")

                result_message = (
                    "Your predicted productivity category is Low. "
                    "Start with one manageable improvement to your daily routine."
                )


            else:

                st.info(
                    f"Predicted category: {prediction_label}"
                )

                result_message = (
                    f"The model returned the category {prediction_label}."
                )


            st.markdown(
                f'<div class="result-panel">{result_message}</div>',
                unsafe_allow_html=True
            )


            # ---------------- DAILY SNAPSHOT ----------------

            st.markdown(
                '<div class="center-sub">'
                'Your daily snapshot</div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Study hours",
                f"{study_hours:g} h"
            )

            c2.metric(
                "Sleep hours",
                f"{sleep_hours:g} h"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Stress level",
                f"{stress_level}/10"
            )

            c2.metric(
                "Focus score",
                f"{focus_score}/100"
            )

            c3.metric(
                "Screen time",
                f"{screen_time:g} h"
            )


            # ---------------- PRACTICAL NEXT STEPS ----------------

            st.markdown(
                '<div class="center-sub">'
                'Your practical next steps</div>',
                unsafe_allow_html=True
            )

            tips = []

            if study_hours < 2:

                tips.append(
                    "Try starting with one focused 25-minute study session."
                )

            elif study_hours >= 8:

                tips.append(
                    "Plan short breaks between study sessions to avoid fatigue."
                )


            if sleep_hours < 7:

                tips.append(
                    "Consider making more room for sleep and keeping a regular bedtime."
                )


            if stress_level >= 7:

                tips.append(
                    "Break large tasks into smaller steps and take short recovery breaks."
                )


            if focus_score <= 30:

                tips.append(
                    "Study in a quieter place and keep unrelated notifications off."
                )


            if screen_time >= 6:

                tips.append(
                    "Try limiting non-study screen use during your planned study time."
                )


            if not tips:

                tips.append(
                    "Keep your routine consistent and review your study habits regularly."
                )


            for tip in tips:

                st.markdown(
                    f'<div class="tip-card">↗ &nbsp; {tip}</div>',
                    unsafe_allow_html=True
                )


            # ---------------- DOWNLOAD CSV REPORT ----------------

            report = input_data.copy()

            report.insert(
                0,
                "predicted_productivity_category",
                prediction_label
            )

            report.insert(
                0,
                "generated_at",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )


            st.markdown(
                '<div class="center-sub">'
                'Download your result</div>',
                unsafe_allow_html=True
            )

            csv_bytes = report.to_csv(
                index=False
            ).encode("utf-8-sig")


            st.download_button(
                "↓  Download productivity report (CSV)",
                data=csv_bytes,
                file_name="focusiq_productivity_report.csv",
                mime="text/csv",
                use_container_width=True
            )


        except Exception as error:

            st.error(
                "Prediction could not be completed."
            )

            st.code(str(error))

            st.info(
                "Verify that the saved scaler and model were trained with "
                "the same five features, in the same order, and that the "
                "model expects scaled input."
            )


# -------------------- FOOTER --------------------

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer">'
    'FOCUSIQ / STUDENT FOCUS LAB<br>'
    'Machine Learning Based Student Productivity Prediction'
    '</div>',
    unsafe_allow_html=True
)
