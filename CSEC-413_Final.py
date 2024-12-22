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
        # Configure Streamlit page
        st.set_page_config(
            page_title="Modeling & Simulation Workflow", 
            page_icon="🔬",
            layout="wide"
        )
        
        # Initialize session state for data storage
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        
    def introduction_page(self):
        """
        Introduction and project overview
        """
        st.title("🧪 Modeling and Simulation Project")
        
        st.header("Introduction")
        st.markdown("""
        Welcome to the **Modeling and Simulation Workflow Application**. This tool demonstrates the end-to-end 
        process of data-driven modeling and simulation, from generating synthetic data to evaluating predictive models.
        """)
        
        st.header("Workflow Overview")
        st.markdown("""
        The application consists of the following steps:
        1. **Data Generation**: Create synthetic datasets with customizable features.
        2. **Exploratory Data Analysis (EDA)**: Visualize and understand the data.
        3. **Modeling**: Train machine learning models to predict target variables.
        4. **Simulation**: Analyze scenarios and their impacts on predictions.
        5. **Evaluation**: Assess model performance and interpret results.
        """)
        
    def data_generation_page(self):
        """
        Synthetic data generation page
        """
        st.header("🔢 Data Generation")
        st.markdown("""
        Generate synthetic datasets with customizable features and noise levels.
        Use the sidebar to configure data generation parameters.
        """)
        
        # Sidebar for parameter configuration
        st.sidebar.header("Data Generation Parameters")
        predefined_features = st.sidebar.multiselect(
            "Predefined Features", 
            ["Temperature", "Pressure", "Humidity", "Wind Speed", "Altitude"], 
            default=["Temperature", "Pressure", "Humidity"]
        )
        
        custom_features = st.sidebar.text_area(
            "Custom Features (comma-separated)", 
            help="Add custom features like 'Solar Radiation, Soil Moisture'."
        )
        custom_features = [feature.strip() for feature in custom_features.split(",") if feature.strip()]
        
        features = predefined_features + custom_features
        n_samples = st.sidebar.slider("Number of Samples", 100, 10000, 1000)
        noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1, step=0.01)
        
        if st.sidebar.button("Generate Data"):
            data = self.generate_synthetic_data(features, n_samples, noise_level)
            st.session_state.generated_data = data
            st.success("Data generated successfully!")
            st.subheader("Generated Data")
            st.dataframe(data)
        
        if st.session_state.generated_data is not None:
            st.subheader("Current Dataset")
            st.dataframe(st.session_state.generated_data)
    
    def generate_synthetic_data(self, features, n_samples, noise_level):
        """
        Generate synthetic data with realistic relationships
        """
        np.random.seed(42)
        data = pd.DataFrame()
        
        for feature in features:
            if feature == "Temperature":
                base_temp = np.random.normal(25, 5, n_samples)
                data["Temperature"] = base_temp
            elif feature == "Pressure":
                data["Pressure"] = np.random.normal(1000, 50, n_samples)
            elif feature == "Humidity":
                data["Humidity"] = np.random.uniform(0, 100, n_samples)
            elif feature == "Wind Speed":
                data["Wind Speed"] = np.abs(np.random.normal(5, 2, n_samples))
            elif feature == "Altitude":
                data["Altitude"] = np.random.gamma(5, 100, n_samples)
            else:
                data[feature] = np.random.normal(0, 1, n_samples)
        
        # Target variable with realistic relationships
        if features:
            target = np.zeros(n_samples)
            weights = {feature: 1 / len(features) for feature in features}
            for feature in features:
                if feature in data.columns:
                    normalized_feature = (data[feature] - data[feature].mean()) / data[feature].std()
                    target += weights[feature] * normalized_feature
            data["Target"] = target + np.random.normal(0, noise_level, n_samples)
        
        return data
    
    def exploratory_analysis_page(self):
        """
        Exploratory Data Analysis (EDA) page
        """
        st.header("📊 Exploratory Data Analysis")
        
        if st.session_state.generated_data is None:
            st.warning("Please generate data first.")
            return
        
        data = st.session_state.generated_data
        
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())
        
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        st.pyplot(plt)
        
        st.subheader("Feature Distributions")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3 * len(numeric_cols)))
        for i, col in enumerate(numeric_cols):
            sns.histplot(data[col], kde=True, ax=axes[i])
            axes[i].set_title(f"{col} Distribution")
        plt.tight_layout()
        st.pyplot(fig)
    
    def modeling_page(self):
        """
        Modeling and evaluation page
        """
        st.header("🤖 Modeling and Evaluation")
        
        if st.session_state.generated_data is None:
            st.warning("Please generate data first.")
            return
        
        data = st.session_state.generated_data
        X = data.drop("Target", axis=1)
        y = data["Target"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)
        
        st.subheader("Model Performance")
        st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.4f}")
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": rf_model.feature_importances_
        }).sort_values("importance", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feature_importance)
        plt.title("Feature Importance")
        st.pyplot(plt)
    
    def simulation_page(self):
        """
        Scenario simulation and analysis page
        """
        st.header("🔮 Simulation and Scenario Analysis")
        if st.session_state.generated_data is None:
            st.warning("Please generate data first.")
            return
        st.info("Use the sliders to simulate different scenarios and see their impact.")
    
    def main(self):
        """
        Main application workflow
        """
        pages = {
            "🏠 Introduction": self.introduction_page,
            "🔢 Data Generation": self.data_generation_page,
            "📊 Exploratory Analysis": self.exploratory_analysis_page,
            "🤖 Modeling": self.modeling_page,
            "🔮 Simulation": self.simulation_page,
        }
        
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        pages[selection]()

# Run the application
if __name__ == "__main__":
    app = ModelingSimulationApp()
    app.main()
