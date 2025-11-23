"""
Modern Streamlit Web Application for Disease Prediction (2023)
Replaces the old Tkinter GUI with a professional web interface
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px

# Add to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import DiseaseDataLoader
from src.models.classical_models import ClassicalMLModels
from src.models.deep_learning_models import DeepLearningTrainer
from configs.config import MODELS_DIR

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    data_loader = DiseaseDataLoader()

    # Load classical models
    try:
        classical_models = ClassicalMLModels()
        for model_name in ['decision_tree', 'random_forest', 'naive_bayes']:
            model_path = MODELS_DIR / f"{model_name}_model.pkl"
            if model_path.exists():
                classical_models.load_model(model_name, model_path)
                models[model_name] = classical_models.models[model_name]
    except Exception as e:
        st.warning(f"Could not load classical models: {e}")

    # Load deep learning models
    try:
        # Load training data to get dimensions
        train_df, _ = data_loader.load_data()
        input_dim = len([col for col in train_df.columns if col != 'prognosis'])

        # Get number of diseases
        output_dim = train_df['prognosis'].nunique()

        # Neural Network
        nn_path = MODELS_DIR / "neural_network_model.pth"
        if nn_path.exists():
            nn_trainer = DeepLearningTrainer('neural_network', input_dim, output_dim)
            nn_trainer.load_model(nn_path)
            models['neural_network'] = nn_trainer

        # Transformer
        transformer_path = MODELS_DIR / "transformer_model.pth"
        if transformer_path.exists():
            transformer_trainer = DeepLearningTrainer('transformer', input_dim, output_dim)
            transformer_trainer.load_model(transformer_path)
            models['transformer'] = transformer_trainer
    except Exception as e:
        st.warning(f"Could not load deep learning models: {e}")

    return models, data_loader


def get_symptom_list():
    """Get list of all symptoms"""
    symptoms = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
    'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
    'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
    'scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze']

    return sorted(symptoms)


def make_prediction(symptoms, model_name, models, data_loader):
    """Make prediction using selected model"""
    # Prepare symptoms
    X = data_loader.preprocess_symptoms(symptoms)

    # Get model
    model = models[model_name]

    # Make prediction
    if model_name in ['neural_network', 'transformer']:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
    else:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None

    # Get disease name
    disease_name = data_loader.disease_names[prediction]

    return disease_name, probabilities, prediction


def main():
    """Main application"""

    # Header
    st.markdown('<h1 class="main-header">🏥 Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Advanced Machine Learning & Deep Learning (2023)</p>', unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models..."):
        models, data_loader = load_models()

    if not models:
        st.error("⚠️ No models found! Please train models first by running: python src/train_and_evaluate.py")
        return

    st.success(f"✅ Loaded {len(models)} models successfully!")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    model_options = {
        'decision_tree': 'Decision Tree (Classical)',
        'random_forest': 'Random Forest (Classical)',
        'naive_bayes': 'Naive Bayes (Classical)',
        'neural_network': 'Deep Neural Network (2023)',
        'transformer': 'Transformer (2023 - Latest)'
    }

    available_models = {k: v for k, v in model_options.items() if k in models}

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(available_models.keys()),
        format_func=lambda x: available_models[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")

    if selected_model in ['neural_network', 'transformer']:
        st.sidebar.info(f"🔥 **{available_models[selected_model]}**\n\nState-of-the-art deep learning model using modern 2023 architecture.")
    else:
        st.sidebar.info(f"📊 **{available_models[selected_model]}**\n\nClassical machine learning model with optimized hyperparameters.")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📝 Patient Information</h2>', unsafe_allow_html=True)

        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")

        st.markdown("### Select Symptoms")
        st.markdown("*Select up to 5 symptoms the patient is experiencing*")

        symptoms_list = get_symptom_list()

        # Multi-select for symptoms
        selected_symptoms = st.multiselect(
            "Search and select symptoms:",
            options=symptoms_list,
            max_selections=5,
            help="Start typing to search for symptoms"
        )

        # Quick symptom selection
        st.markdown("#### Common Symptom Groups")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("🤒 Fever Related"):
                selected_symptoms = ['mild_fever', 'fatigue', 'headache', 'muscle_pain'][:5]
                st.rerun()

        with col_b:
            if st.button("🤢 Digestive Issues"):
                selected_symptoms = ['abdominal_pain', 'diarrhoea', 'nausea', 'vomiting'][:5]
                st.rerun()

        with col_c:
            if st.button("😮‍💨 Respiratory"):
                selected_symptoms = ['cough', 'chest_pain', 'throat_irritation', 'congestion'][:5]
                st.rerun()

        st.markdown("---")

        # Display selected symptoms
        if selected_symptoms:
            st.markdown("### Selected Symptoms:")
            for i, symptom in enumerate(selected_symptoms, 1):
                st.markdown(f"{i}. **{symptom.replace('_', ' ').title()}**")

            # Predict button
            if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
                if len(selected_symptoms) == 0:
                    st.warning("Please select at least one symptom!")
                else:
                    with st.spinner("Analyzing symptoms..."):
                        disease, probabilities, pred_idx = make_prediction(
                            selected_symptoms, selected_model, models, data_loader
                        )

                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Prediction Results")

                    if patient_name:
                        st.markdown(f"**Patient:** {patient_name}")

                    st.markdown(f"**Model Used:** {available_models[selected_model]}")
                    st.markdown(f"### Predicted Disease: **{disease}**")

                    # Confidence scores
                    if probabilities is not None:
                        confidence = probabilities[pred_idx] * 100
                        st.markdown(f"**Confidence:** {confidence:.2f}%")

                        # Progress bar
                        st.progress(probabilities[pred_idx])

                        # Top 5 predictions
                        st.markdown("#### Top 5 Possible Diseases:")
                        top_5_idx = np.argsort(probabilities)[-5:][::-1]

                        for idx in top_5_idx:
                            disease_name = data_loader.disease_names[idx]
                            prob = probabilities[idx] * 100
                            st.markdown(f"- **{disease_name}**: {prob:.2f}%")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Disclaimer
                    st.warning("⚠️ **Disclaimer:** This prediction is for informational purposes only. Please consult a healthcare professional for proper diagnosis and treatment.")

        else:
            st.info("👆 Please select symptoms from the list above")

    with col2:
        st.markdown('<h2 class="sub-header">📊 System Statistics</h2>', unsafe_allow_html=True)

        # Statistics
        st.metric("Total Models Available", len(models))
        st.metric("Total Symptoms", len(symptoms_list))

        if data_loader.disease_names:
            st.metric("Total Diseases", len(data_loader.disease_names))

        st.markdown("---")

        st.markdown("### 🚀 Model Types")
        for model_key, model_name in available_models.items():
            if model_key in ['neural_network', 'transformer']:
                st.markdown(f"✅ **{model_name}**")
            else:
                st.markdown(f"📊 **{model_name}**")

        st.markdown("---")

        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses state-of-the-art machine learning and deep learning models to predict diseases based on symptoms.

        **Features:**
        - 5 Advanced ML Models
        - Neural Networks & Transformers
        - Real-time Predictions
        - Confidence Scores
        - Modern 2023 Architecture
        """)


if __name__ == "__main__":
    main()
