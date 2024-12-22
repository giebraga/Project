import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ModelingSimulationApp:
    def __init__(self):
        """
        Initialize the Streamlit application with configurations and session state.
        """
        st.set_page_config(
            page_title="Modeling & Simulation Workflow",
            page_icon="🔬",
            layout="wide"
        )
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None

    def introduction_page(self):
        """
        Display the project introduction and overview.
        """
        st.title("🧪 Modeling and Simulation Project")
        st.header("Project Introduction")
        st.markdown("""
        This interactive application demonstrates a comprehensive workflow for 
        **Modeling and Simulation using Python**. Explore data science techniques 
        through synthetic data generation, analysis, modeling, and simulation.
        """)
        st.header("Project Workflow Steps")
        steps = [
            "**Data Generation**: Create synthetic data mimicking real-world scenarios.",
            "**Exploratory Data Analysis (EDA)**: Investigate data characteristics.",
            "**Modeling**: Apply machine learning techniques for prediction.",
            "**Simulation**: Generate predictive outcomes and explore scenarios.",
            "**Evaluation**: Assess model performance and reliability."
        ]
        for step in steps:
            st.markdown(f"- {step}")
        st.info("🔍 Use the sidebar to navigate through each step of the workflow.")

    def data_generation_page(self):
        """
        Allow users to generate synthetic data interactively.
        """
        st.header("🔢 Data Generation")
        st.markdown("""
        Generate synthetic data with controllable properties to simulate 
        real-world scenarios. Customize the parameters below.
        """)
        st.sidebar.header("Data Generation Parameters")
        
        # Feature selection
        predefined_features = st.sidebar.multiselect(
            "Select Predefined Features", 
            ["Temperature", "Pressure", "Humidity", "Wind Speed", "Altitude"],
            default=["Temperature", "Pressure", "Humidity"]
        )
        custom_features = st.sidebar.text_area(
            "Define Custom Features (comma-separated)",
            help="Enter custom feature names, e.g., 'Solar Radiation, Soil Moisture'"
        )
        custom_features = [feature.strip() for feature in custom_features.split(",") if feature.strip()]
        features = predefined_features + custom_features
        
        # Data generation settings
        n_samples = st.sidebar.slider("Number of Samples", 100, 10000, 1000)
        noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1, step=0.01)

        # Generate data
        if st.sidebar.button("Generate Data"):
            data = self.generate_synthetic_data(features, n_samples, noise_level)
            st.session_state.generated_data = data
            st.success("Data generated successfully!")
            st.dataframe(data)

        # Display existing data
        if st.session_state.generated_data is not None:
            st.subheader("Current Dataset")
            st.dataframe(st.session_state.generated_data)

    def generate_synthetic_data(self, features, n_samples, noise_level):
        """
        Generate synthetic data with specified features and realistic correlations.
        """
        np.random.seed(42)
        data = pd.DataFrame()

        # Generate data for predefined and custom features
        for feature in features:
            if feature == "Altitude":
                data["Altitude"] = np.random.gamma(5, 100, n_samples)
            elif feature == "Temperature":
                base_temp = np.random.normal(25, 5, n_samples)
                if "Altitude" in data.columns:
                    data["Temperature"] = base_temp - (data["Altitude"] * 0.0065)
                else:
                    data["Temperature"] = base_temp
            elif feature == "Pressure":
                if "Altitude" in data.columns:
                    standard_pressure = 1013.25
                    data["Pressure"] = standard_pressure * np.exp(-0.0001 * data["Altitude"]) + np.random.normal(0, 20, n_samples)
                else:
                    data["Pressure"] = np.random.normal(1000, 50, n_samples)
            elif feature == "Humidity":
                if "Temperature" in data.columns:
                    data["Humidity"] = np.clip(100 - (data["Temperature"] * 2) + np.random.normal(50, 10, n_samples), 0, 100)
                else:
                    data["Humidity"] = np.clip(np.random.normal(60, 15, n_samples), 0, 100)
            elif feature == "Wind Speed":
                base_wind = np.abs(np.random.normal(5, 2, n_samples))
                if "Altitude" in data.columns:
                    base_wind += 0.005 * data["Altitude"]
                data["Wind Speed"] = np.clip(base_wind, 0, 30)
            else:
                data[feature] = np.random.normal(0, 1, n_samples)

        # Create a target variable with realistic relationships
        if features:
            target = np.zeros(n_samples)
            weights = {feature: 1/len(features) for feature in features}
            for feature in features:
                if feature in data.columns:
                    normalized_feature = (data[feature] - data[feature].mean()) / data[feature].std()
                    target += weights[feature] * normalized_feature
            data['Target'] = target + np.random.normal(0, noise_level, n_samples)
        return data

    def exploratory_analysis_page(self):
        """
        Perform Exploratory Data Analysis (EDA) with visualizations.
        """
        st.header("📊 Exploratory Data Analysis")
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        data = st.session_state.generated_data
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)
        st.subheader("Feature Distributions")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3*len(numeric_cols)))
        for i, col in enumerate(numeric_cols):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'{col} Distribution')
        plt.tight_layout()
        st.pyplot(fig)

    def modeling_page(self):
        """
        Build and evaluate a predictive model.
        """
        st.header("🤖 Modeling and Simulation")
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        data = st.session_state.generated_data
        X = data.drop('Target', axis=1)
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)
        st.subheader("Model Performance")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("R² Score", f"{r2:.4f}")
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        st.pyplot(plt)

    def main(self):
        """
        Main application workflow.
        """
        pages = {
            "🏠 Introduction": self.introduction_page,
            "🔢 Data Generation": self.data_generation_page,
            "📊 Exploratory Analysis": self.exploratory_analysis_page,
            "🤖 Modeling": self.modeling_page
        }
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        pages[selection]()

if __name__ == "__main__":
    app = ModelingSimulationApp()
    app.main()
