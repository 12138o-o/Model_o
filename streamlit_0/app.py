from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to Agg (headless mode)
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Paths and cached loaders
# --------------------------------------------------------------------------------------

# Use data from streamlit directory
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
META_DIR = PROJECT_ROOT.parent / "data"  # Keep access to variable description files


@st.cache_resource(show_spinner=False)
def load_model() -> object:
    model_path = MODEL_DIR / "stacking_classifier_model_optimized_v3.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_training_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load scaled training/test frames for feature order and SHAP background.

    Read with index_col=0 so the first column (saved index) is not treated as a feature.
    """
    x_train_scaled = pd.read_csv(DATA_DIR / "X_train.csv", index_col=0)
    x_test_scaled = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
    return x_train_scaled, x_test_scaled


@st.cache_data(show_spinner=False)
def load_external_validation_data() -> Optional[pd.DataFrame]:
    """Load external validation (China) scaled data if available."""
    charls_path = DATA_DIR / "X_charls_scaled.csv"
    if charls_path.exists():
        return pd.read_csv(charls_path, index_col=0)
    return None


@st.cache_data(show_spinner=False)
def load_variable_help() -> Dict[str, str]:
    """Parse the variable description xlsx to extract the text inside parentheses.

    Returns mapping: column_name -> help_text
    """
    # First try to read Variable.xlsx from streamlit directory
    meta_path = DATA_DIR / "Variable .xlsx"
    if not meta_path.exists():
        meta_path = META_DIR / "Variable .xlsx"
    
    if not meta_path.exists():
        # If variable description file is not found, provide default descriptions
        default_help = {
            "Nationality": "Nationality (USA/China)",
            "Gender": "Gender (0=Female, 1=Male)",
            "Neck_pain": "Neck pain (0=No, 1=Yes)",
            "Arthritis": "Arthritis (0=No, 1=Yes)",
            "Self_perceived_health_status": "Self perceived health status (1=Poor, 2=Fair, 3=Good)",
            "Lung_disease": "Lung disease (0=No, 1=Yes)",
            "Waist_circumference": "Waist circumference (cm)"
        }
        return default_help
    
    try:
        # Try to read Excel file
        rows = pd.read_excel(meta_path, header=None)
        # First row is header, extract variable names and descriptions from second row onwards
        if len(rows) > 1:
            variable_dict = {
                "Nationality": "Nationality (USA/China)",
                "Gender": "Gender (0=Female, 1=Male)"
            }
            for idx in range(1, len(rows)):
                var_name = str(rows.iloc[idx, 0]).strip()
                var_desc = str(rows.iloc[idx, 1]).strip() if len(rows.columns) > 1 else ""
                variable_dict[var_name] = var_desc
            return variable_dict
    except Exception as e:
        print(f"Failed to read Variable.xlsx: {e}")
    
    # If failed, return default values
    default_help = {
        "Nationality": "Nationality (USA/China)",
        "Gender": "Gender (0=Female, 1=Male)",
        "Neck_pain": "Neck pain (0=No, 1=Yes)",
        "Arthritis": "Arthritis (0=No, 1=Yes)",
        "Self_perceived_health_status": "Self perceived health status (1=Poor, 2=Fair, 3=Good)",
        "Lung_disease": "Lung disease (0=No, 1=Yes)",
        "Waist_circumference": "Waist circumference (cm)"
    }
    return default_help


def build_numeric_scaler(x_train_scaled: pd.DataFrame) -> Optional[StandardScaler]:
    """Create a scaler for Waist_circumference (the only continuous variable).

    Priority of sources for mean/std:
    1) data/X_train_notscaled.csv (if present with the expected columns)
    2) Approximate from known ranges using scaled min/max (last resort)
    """
    target_cols = ["Waist_circumference"]

    # 1) Preferred: explicit unscaled training data
    unscaled_path = DATA_DIR / "X_train_notscaled.csv"
    if unscaled_path.exists():
        df_unscaled = pd.read_csv(unscaled_path, index_col=0)
        if all(col in df_unscaled.columns for col in target_cols):
            scaler = StandardScaler().fit(df_unscaled[target_cols])
            return scaler

    # 2) Last resort: approximate from scaled min/max and known raw ranges
    # This is a coarse approximation if raw means/std are not available.
    if set(target_cols).issubset(x_train_scaled.columns):
        z = x_train_scaled[target_cols].copy()
        approx_mean = np.array([105.0])  # typical default for waist circumference
        # sigma derived from span if possible
        try:
            z_span = (z.max() - z.min()).values
            wc_sigma = 30.0 / max(z_span[0], 1e-6)  # waist circumference range ~70..140 -> span ~70
            sigma = np.array([wc_sigma])
            scaler = StandardScaler()
            scaler.mean_ = approx_mean
            scaler.scale_ = sigma
            scaler.var_ = sigma ** 2
            scaler.n_features_in_ = 1
            scaler.feature_names_in_ = np.array(target_cols)
            return scaler
        except Exception:
            return None
    return None


def get_waist_threshold(nationality: str, gender: int) -> Tuple[float, str]:
    """Get waist circumference threshold for different nationalities and genders
    
    Args:
        nationality: "USA" or "China"
        gender: 0=Female, 1=Male
        
    Returns:
        (threshold, message): threshold value and description
    """
    if nationality == "USA":
        if gender == 1:  # Male
            return 102.0, "USA Male: ≥102 cm indicates abdominal obesity"
        else:  # Female
            return 88.0, "USA Female: ≥88 cm indicates abdominal obesity"
    else:  # China
        if gender == 1:  # Male
            return 90.0, "China Male: ≥90 cm indicates abdominal obesity"
        else:  # Female
            return 85.0, "China Female: ≥85 cm indicates abdominal obesity"


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model: object, x_train_scaled: pd.DataFrame):
    """Create and cache a SHAP explainer for the model with optimized background."""
    # Use k-means sampling to reduce background data volume and improve speed
    background_size = min(1000, len(x_train_scaled))  # Limit background sample size
    background = shap.kmeans(x_train_scaled, background_size)
    return shap.KernelExplainer(lambda X: _model.predict_proba(X)[:, 1], background)



def render_title():
    st.set_page_config(
        page_title="Predictors of low back pain risk in middle-aged and elderly people with abdominal obesity",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Set page background to light blue, other areas to off-white
    st.markdown("""
        <style>
        .stApp {
            background-color: #E6F3FF;
        }
        .stForm {
            background-color: #FEFDFB;
            padding: 20px;
            border-radius: 10px;
        }
        /* Main content area */
        .main .block-container {
            background-color: #FEFDFB;
            padding: 2rem;
            border-radius: 10px;
        }
        /* Keep input boxes with original Streamlit default style */
        /* Expander background */
        .streamlit-expanderHeader {
            background-color: #FEFDFB;
        }
        /* Change metric container background to light blue */
        div[data-testid="stMetric"] {
            background-color: #E6F3FF !important;
            padding: 10px;
            border-radius: 8px;
        }
        /* Metric label and value background */
        div[data-testid="stMetricLabel"] {
            background-color: transparent !important;
        }
        div[data-testid="stMetricValue"] {
            background-color: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("🔍 Predictors of low back pain risk in middle-aged and elderly people with abdominal obesity")
    st.caption(
        "Enter patient characteristics to estimate risk and explain predictions through SHAP waterfall plots."
    )


def main():
    render_title()

    try:
        # Load assets
        model = load_model()
        x_train_scaled, x_test_scaled = load_training_frames()
        x_charls_scaled = load_external_validation_data()
        variable_help = load_variable_help()
        scaler = build_numeric_scaler(x_train_scaled)

        # Display data loading status
        if x_charls_scaled is not None:
            st.sidebar.success(f"✅ External validation data loaded: {len(x_charls_scaled)} samples (China)")
        else:
            st.sidebar.info("ℹ️ External validation data not found. Using USA training data only.")

        feature_names: List[str] = list(x_train_scaled.columns)
        if len(feature_names) != 5:
            st.warning(
                f"Detected {len(feature_names)} features in training data, but expected 5. "
                "Please ensure X_train.csv contains the correct 5 columns."
            )

        # ==================== Step 1: Population Screening ====================
        with st.form("screening_form"):
            st.subheader("Patient Information Input")
            st.markdown("### 📋 Step 1: Population Screening")
            
            # First row: Age group
            age_group = st.selectbox(
                "Age Group",
                options=["≥45 years (Middle-aged and elderly)", "<45 years (Younger adults)"],
                index=0,
                help="This model is designed for middle-aged and elderly people (≥45 years). Predictions for younger adults may not be reliable."
            )
            is_middle_aged = age_group.startswith("≥45")
            
            # Second row: Nationality and gender
            col_demo1, col_demo2 = st.columns(2)
            with col_demo1:
                nationality = st.selectbox(
                    "Nationality",
                    options=["USA", "China"],
                    index=0,
                    help=variable_help.get("Nationality", "Select patient's nationality")
                )
            with col_demo2:
                gender = st.selectbox(
                    "Gender",
                    options=["Female (0)", "Male (1)"],
                    index=0,
                    help=variable_help.get("Gender", "Gender (0=Female, 1=Male)")
                )
                gender_value = 0 if gender == "Female (0)" else 1
            
            # Calculate waist circumference threshold (for subsequent validation and help text)
            threshold, threshold_msg = get_waist_threshold(nationality, gender_value)
            
            # Important note
            st.info("""
ℹ️ **Important Note:**
- **Age Group and Nationality/Gender** are used only for **population screening** and **waist circumference threshold determination**.
- These selections **do NOT affect the prediction probability** itself.
- The model predicts based solely on the 5 clinical features (neck pain, arthritis, health status, lung disease, and waist circumference).
- Age and nationality help ensure the model is applied to the appropriate population.
            """)
            
            # Display age applicability warning
            if not is_middle_aged:
                st.warning("⚠️ **Warning**: Age <45 years selected. You will NOT be able to make predictions. Please select ≥45 years to continue.")
            
            # Display waist circumference standards
            st.info(f"""
📏 **Abdominal Obesity Standards:**
- 🇺🇸 USA: Male ≥102 cm, Female ≥88 cm
- 🇨🇳 China: Male ≥90 cm, Female ≥85 cm

**Your current threshold**: {threshold} cm for {nationality} {gender.split('(')[0].strip()}
            """)
            
            # Step 1 confirmation button
            st.markdown("---")
            confirm_eligibility = st.form_submit_button(
                "✅ Confirm Eligibility and Continue" if is_middle_aged else "❌ Age <45 - Cannot Continue",
                disabled=not is_middle_aged,
                use_container_width=True
            )
        
        # Use session state to save confirmation status
        if confirm_eligibility and is_middle_aged:
            st.session_state.eligibility_confirmed = True
            st.session_state.nationality = nationality
            st.session_state.gender = gender
            st.session_state.gender_value = gender_value
            st.session_state.threshold = threshold
            st.session_state.age_group = age_group
        
        # ==================== Step 2: Clinical Features Input ====================
        # Only display Step 2 after eligibility confirmation
        if st.session_state.get('eligibility_confirmed', False):
            # Read screening information from session state
            nationality = st.session_state.nationality
            gender = st.session_state.gender
            gender_value = st.session_state.gender_value
            threshold = st.session_state.threshold
            age_group = st.session_state.age_group
            
            st.markdown("---")
            with st.form("feature_input_form"):
                st.markdown("### 🩺 Step 2: Clinical Features Input")
                st.caption("Enter the patient's clinical characteristics below.")
                
                # Display confirmed population information
                st.success(f"✅ **Confirmed**: {age_group} | {nationality} {gender.split('(')[0].strip()} | Waist threshold: {threshold} cm")
                
                # Other feature inputs
                cols = st.columns(2)

                # Utility for selectboxes
                def sb(idx: int, label: str, options: Dict[str, int], default_key: str) -> int:
                    help_text = variable_help.get(label, "")
                    choice = cols[idx % 2].selectbox(
                        f"{label}",
                        options=list(options.keys()),
                        index=list(options.keys()).index(default_key),
                        help=help_text,
                    )
                    return options[choice]

                # Inputs - 5 features
                neck_pain = sb(0, "Neck_pain", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
                arthritis = sb(1, "Arthritis", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
                selfhea = sb(
                    0,
                    "Self_perceived_health_status",
                    {"Poor (1)": 1, "Fair (2)": 2, "Good (3)": 3},
                    "Fair (2)",
                )
                lung_disease = sb(1, "Lung_disease", {"No (0)": 0, "Yes (1)": 1}, "No (0)")
                
                waist_circumference_raw = cols[0].number_input(
                    "Waist_circumference",
                    min_value=60.0,
                    max_value=170.0,
                    value=105.0,
                    step=0.1,
                    help=variable_help.get("Waist_circumference", f"Waist circumference (cm), threshold: {threshold} cm"),
                )
                
                # Add unified comments above the prediction button
                st.markdown("---")
                st.markdown("**Variable Definitions:**", help="Click to view detailed explanations")
                with st.expander("📋 Variable Definitions"):
                    st.markdown(f"""
                    **Neck_pain**: Do you have neck pain? (0=No, 1=Yes)
                    
                    **Arthritis**: Has a doctor ever told you that you have arthritis? (0=No, 1=Yes)
                    
                    **Self_perceived_health_status**: How do you rate your overall health? (1=Poor, 2=Fair, 3=Good)
                    
                    **Lung_disease**: Has a doctor ever told you that you have lung disease? (0=No, 1=Yes)
                    
                    **Waist_circumference**: Waist circumference measured in centimeters (cm)
                    - Current threshold: {threshold} cm (based on {nationality} {gender})
                    - Model is designed for individuals with abdominal obesity
                    """)
                
                col_btn1, col_btn2 = st.columns([3, 1])
                with col_btn1:
                    submitted = st.form_submit_button("🔮 Start Prediction", use_container_width=True)
                with col_btn2:
                    reset = st.form_submit_button("🔄 Reset", use_container_width=True)
                
                if reset:
                    st.session_state.eligibility_confirmed = False
                    st.rerun()
        else:
            # If eligibility not confirmed, display prompt
            st.info("👆 **Please complete Step 1: Population Screening and click 'Confirm Eligibility and Continue' to proceed.**")
            # Set default values to avoid errors in subsequent code
            submitted = False
            neck_pain = 0
            arthritis = 0
            selfhea = 2
            lung_disease = 0
            waist_circumference_raw = 105.0

        # Prepare the single-row input in the exact feature order
        user_raw: Dict[str, float] = {
            "Neck_pain": neck_pain,
            "Arthritis": arthritis,
            "Self_perceived_health_status": selfhea,
            "Lung_disease": lung_disease,
            "Waist_circumference": float(waist_circumference_raw),
        }

        # Create dataframe and scale numeric fields to match the model's expectations
        sample_df_unscaled = pd.DataFrame([user_raw])
        # Align columns in the correct order
        sample_df_unscaled = sample_df_unscaled[[c for c in feature_names]]

        # Transform numerics using scaler learned from training (unscaled -> scaled)
        numerics = ["Waist_circumference"]  # Only Waist_circumference needs scaling
        scaled_values = sample_df_unscaled.copy()
        if numerics and scaler is not None:
            scaled_values[numerics] = scaler.transform(sample_df_unscaled[numerics])
        else:
            st.warning(
                "Automatic standardization for Waist_circumference is not available. "
                "Please enter standardized values (z-scores) directly."
            )

        if submitted:
            # First check if age meets requirements
            if not is_middle_aged:
                st.error("❌ **Cannot Make Prediction**")
                st.warning(
                    "⚠️ **Age group (<45 years) does not meet the model requirements.**\n\n"
                    "This model is specifically designed for **middle-aged and elderly people (≥45 years)**. "
                    "The training and validation data only included participants aged 45 years and above.\n\n"
                    "**Why age matters:**\n"
                    "- Risk factors for low back pain differ significantly between age groups\n"
                    "- The model was trained on middle-aged and elderly populations\n"
                    "- Predictions for younger adults (<45 years) may not be reliable or accurate\n\n"
                    "**Recommendation:**\n"
                    "Please select **'≥45 years (Middle-aged and elderly)'** if the patient is 45 years or older.\n\n"
                    "For younger adults, please consult appropriate clinical guidelines or models designed for that age group."
                )
                st.stop()  # Stop execution, do not display prediction results
            
            # Then check if waist circumference meets abdominal obesity criteria
            if waist_circumference_raw < threshold:
                st.error("❌ **Cannot Make Prediction**")
                st.warning(
                    f"⚠️ **Waist circumference ({waist_circumference_raw:.1f} cm) is below "
                    f"the abdominal obesity threshold ({threshold} cm) for {nationality} {gender.split('(')[0].strip()}.**\n\n"
                    f"This model is specifically designed for individuals with abdominal obesity. "
                    f"Predictions for individuals below the threshold may not be reliable.\n\n"
                    f"**Abdominal Obesity Thresholds:**\n"
                    f"- 🇺🇸 USA: Male ≥102 cm, Female ≥88 cm\n"
                    f"- 🇨🇳 China: Male ≥90 cm, Female ≥85 cm\n\n"
                    f"Please ensure the waist circumference meets the appropriate threshold before proceeding."
                )
                st.stop()  # Stop execution, do not display prediction results
            
            # Display confirmation that both age and waist circumference meet criteria
            st.success("✅ **Eligibility Confirmed**")
            st.info(
                f"✓ Age: ≥45 years (Middle-aged and elderly)\n\n"
                f"✓ Waist circumference: {waist_circumference_raw:.1f} cm (≥{threshold} cm for {nationality} {gender.split('(')[0].strip()})"
            )
            
            # Use the most accurate and effective prediction probability method
            try:
                # Directly use the model's predict_proba method
                proba = float(model.predict_proba(scaled_values)[0, 1])
                predicted_class = int(proba >= 0.5)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # Display prediction results
            st.subheader("Prediction Results")
            
            # Display patient information
            st.markdown(f"**Patient Profile**: {nationality} {gender} with waist circumference {waist_circumference_raw:.1f} cm")
            
            # Use more intuitive display method
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", "At Risk" if predicted_class == 1 else "Low Risk")
                st.metric("Risk Probability", f"{proba:.1%}")
            
            with col2:
                if predicted_class == 1:
                    st.error("⚠️ Prediction: Low Back Pain Risk Present")
                else:
                    st.success("✅ Prediction: Low Low Back Pain Risk")
                
                st.info(f"Detailed Probability: At Risk {proba:.1%} | Low Risk {(1.0-proba):.1%}")
            
            # Model applicability note
            if nationality == "China":
                if x_charls_scaled is not None:
                    st.info("ℹ️ **Model Note**: This model was trained on USA population data (NHANES). "
                          f"External validation on {len(x_charls_scaled)} Chinese samples is available for reference. "
                          "SHAP explanation uses USA training data as background.")
                else:
                    st.warning("⚠️ **Model Applicability Note**: This model was trained on USA population data (NHANES). "
                              "Predictions for Chinese population are for reference only and may have reduced accuracy. "
                              "The SHAP explanation uses USA training data as background.")

            # Explain with SHAP Waterfall
            st.subheader("SHAP Feature Importance Explanation")
            with st.spinner("Calculating SHAP explanation..."):
                try:
                    # Use cached SHAP explainer (unified use of training set as background)
                    explainer = get_shap_explainer(model, x_train_scaled)
                    shap_values = explainer(scaled_values)
                    
                    # Waterfall plot centered and taking half width
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        # Ensure matplotlib uses Agg backend (headless mode)
                        plt.switch_backend('Agg')
                        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                        fig = plt.gcf()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)  # Clean up figure to avoid memory leak
                    
                    # Add explanation
                    st.info("💡 SHAP plot shows the contribution of each feature to the prediction result. Positive values indicate increased risk, negative values indicate decreased risk.")
                except Exception as e:
                    st.error(f"SHAP analysis failed: {e}")
                    st.info("Please check if model and data are loaded correctly.")

            st.warning("⚠️ Important Reminder: This tool provides data-driven estimates and should not replace professional medical advice.")

        # Add usage instructions
        with st.expander("📖 Usage Instructions"):
            charls_info = f"\n            - External validation data: {len(x_charls_scaled)} Chinese samples available" if x_charls_scaled is not None else ""
            st.markdown(f"""
            **Model Description:**
            - This model uses machine learning (Stacking Classifier) algorithms to predict low back pain risk in middle-aged and elderly people with abdominal obesity
            - **Target Population**: Middle-aged and elderly individuals (≥45 years) with abdominal obesity
            - Input 5 key clinical features including neck pain, arthritis, self-perceived health status, lung disease, and waist circumference
            - Model was trained on USA population data (NHANES){charls_info}
            
            **Eligibility Criteria:**
            - ✅ **Age**: Must be ≥45 years (middle-aged and elderly)
            - ✅ **Abdominal Obesity**: Must meet waist circumference thresholds
              - 🇺🇸 USA: Male ≥102 cm, Female ≥88 cm
              - 🇨🇳 China: Male ≥90 cm, Female ≥85 cm
            
            **Result Interpretation:**
            - Predicted Class: 0=Low Risk, 1=At Risk
            - Risk Probability: Value between 0-1, closer to 1 indicates higher risk
            - SHAP Plot: Shows the impact of each feature on the prediction result
            
            **Important Notes:**
            - ⚠️ Age <45 years: Predictions will be blocked (model not designed for this group)
            - ⚠️ Below waist threshold: Predictions will be blocked (model requires abdominal obesity)
            - Waist circumference is automatically standardized using USA training data statistics
            - All categorical variables are encoded as numerical values
            - Model predictions for Chinese population are for reference only (trained on USA data)
            - SHAP explanation uses USA training data as background for consistency
            - Actual application requires clinical judgment
            """)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check if data files and model files exist and are in the correct format.")


if __name__ == "__main__":
    main()

