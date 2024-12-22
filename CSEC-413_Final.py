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
        # Initialize session state
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None

    def introduction_page(self):
        """
        Introduction page with project overview and goals.
        """
        st.title("🧪 Modeling and Simulation Project")
        st.markdown("""
        This interactive app showcases a comprehensive **Modeling and Simulation** workflow,
        covering synthetic data generation, exploratory analysis, machine learning, and scenario simulations.
        """)
        st.info("""
        🔍 **Explore the following steps**:
        - **Data Generation**: Simulate realistic datasets.
        - **Exploratory Analysis**: Visualize and understand data properties.
        - **Modeling**: Build predictive models.
        - **Simulation**: Evaluate different scenarios.
        """)

    def data_generation_page(self):
        """
        Page to generate synthetic data.
        """
        st.header("🔢 Data Generation")
        st.sidebar.header("Parameters")
        
        # Feature selection and customization
        features = st.sidebar.multiselect(
            "Select Features",
            ["Temperature", "Pressure", "Humidity", "Wind Speed", "Altitude"],
            default=["Temperature", "Pressure", "Humidity"]
        )
        n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000, step=100)
        noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1, step=0.01)

        # Generate data button
        if st.sidebar.button("Generate Data"):
            data = self.generate_synthetic_data(features, n_samples, noise_level)
            st.session_state.generated_data = data
            st.experimental_rerun()  # Refresh app to display generated data

        # Display data if generated
        if st.session_state.generated_data is not None:
            st.subheader("Generated Synthetic Data")
            st.dataframe(st.session_state.generated_data)

    def generate_synthetic_data(self, features, n_samples, noise_level):
        """
        Generate synthetic data with specified features and noise level.
        """
        np.random.seed(42)
        data = pd.DataFrame()

        if "Altitude" in features:
            data["Altitude"] = np.random.gamma(5, 100, n_samples)
        if "Temperature" in features:
            base_temp = np.random.normal(25, 5, n_samples)
            data["Temperature"] = (
                base_temp - (data["Altitude"] * 0.0065) if "Altitude" in features else base_temp
            )
        if "Pressure" in features:
            data["Pressure"] = (
                1013.25 * np.exp(-0.0001 * data["Altitude"]) if "Altitude" in features else
                base_temp * 0.5 + np.random.normal(1000, 50, n_samples)
            )
        if "Humidity" in features:
            data["Humidity"] = np.clip(100 - (data["Temperature"] * 2) +
                                       np.random.normal(50, 10, n_samples), 0, 100)
        if "Wind Speed" in features:
            data["Wind Speed"] = np.clip(
                np.abs(np.random.normal(5, 2, n_samples)) + 0.005 * data.get("Altitude", 0), 0, 30
            )

        # Add target variable with relationships to features
        weights = {"Temperature": 0.3, "Pressure": 0.2, "Humidity": 0.15, "Wind Speed": 0.15, "Altitude": 0.2}
        data["Target"] = sum(
            weights[feat] * (data[feat] - data[feat].mean()) / data[feat].std() for feat in features
            if feat in data
        ) + np.random.normal(0, noise_level, n_samples)

        return data

    def exploratory_analysis_page(self):
        """
        Perform and visualize exploratory data analysis.
        """
        st.header("📊 Exploratory Data Analysis")
        data = st.session_state.generated_data

        if data is None:
            st.warning("Generate data before proceeding.")
            return

        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())

        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot()

    def modeling_page(self):
        """
        Train and evaluate a predictive model.
        """
        st.header("🤖 Modeling and Simulation")
        data = st.session_state.generated_data

        if data is None:
            st.warning("Generate data before proceeding.")
            return

        # Train-test split
        X, y = data.drop(columns="Target"), data["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        mse, r2 = mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

        st.metric("MSE", f"{mse:.4f}")
        st.metric("R²", f"{r2:.4f}")

        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns, 'Importance': rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        sns.barplot(x="Importance", y="Feature", data=feature_importance)
        st.pyplot()

    def simulation_page(self):
        """
        Interactive simulation of scenarios.
        """
        st.header("🔮 Scenario Simulation")
        data = st.session_state.generated_data

        if data is None:
            st.warning("Generate data before proceeding.")
            return

        st.info("Adjust feature values to simulate different scenarios.")
        # Add interactive sliders for scenario adjustments

    def conclusion_page(self):
        """
        Summarize the project outcomes.
        """
        st.header("🏁 Conclusion")
        st.markdown("""
        Key takeaways:
        - Mastering data generation and exploratory analysis.
        - Implementing machine learning models.
        - Understanding scenario simulations.
        """)
        st.info("🔍 Keep experimenting to improve your skills!")

    def main(self):
        """
        Main workflow with page navigation.
        """
        pages = {
            "🏠 Introduction": self.introduction_page,
            "🔢 Data Generation": self.data_generation_page,
            "📊 Exploratory Analysis": self.exploratory_analysis_page,
            "🤖 Modeling": self.modeling_page,
            "🔮 Simulation": self.simulation_page,
            "🏁 Conclusion": self.conclusion_page,
        }
        st.sidebar.title("Navigation")
        selected_page = st.sidebar.radio("Go to", pages.keys())
        pages[selected_page]()


if __name__ == "__main__":
    app = ModelingSimulationApp()
    app.main()
