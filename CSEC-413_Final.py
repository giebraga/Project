import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

class ModelingSimulationApp:
    def __init__(self):
        # Configure Streamlit page
        st.set_page_config(
            page_title="Modeling & Simulation Workflow", 
            page_icon="🔬",
            layout="wide"
        )
        
        # Initialize session state for data
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        
    def introduction_page(self):
        """
        Project introduction and overview page
        """
        st.title("🧪 Modeling and Simulation Project")
        
        # Introduction section
        st.header("Project Introduction")
        st.markdown("""
        This interactive application demonstrates a comprehensive workflow for 
        **Modeling and Simulation using Python**. The goal is to provide 
        hands-on experience with powerful Python libraries and data science techniques.
        """)
        
        # Project steps overview
        st.header("Project Workflow Steps")
        steps = [
            "**Data Generation**: Create synthetic data mimicking real-world scenarios",
            "**Exploratory Data Analysis (EDA)**: Investigate data characteristics",
            "**Modeling**: Apply appropriate machine learning techniques",
            "**Simulation**: Generate predictive outcomes",
            "**Evaluation**: Assess model performance and reliability"
        ]
        
        for step in steps:
            st.markdown(f"- {step}")
        
        st.info("""
        🔍 This application will guide you through each step of the modeling 
        and simulation process, demonstrating key data science concepts and techniques.
        """)
    
    def data_generation_page(self):
        """
        Interactive data generation section
        """
        st.header("🔢 Data Generation")
        st.markdown("""
        Generate synthetic data with controllable properties to simulate 
        real-world scenarios. Customize data generation parameters below.
        """)
        
        # Sidebar configuration
        st.sidebar.header("Data Generation Parameters")
        
        # Feature configuration
        predefined_features = st.sidebar.multiselect(
            "Select Predefined Features", 
            ["Temperature", "Pressure", "Humidity", "Wind Speed", "Altitude"],
            default=["Temperature", "Pressure", "Humidity"]
        )
        
        # Allow user to define custom features
        custom_features = st.sidebar.text_area(
            "Define Custom Features (comma-separated)", 
            help="Enter custom feature names, e.g., 'Solar Radiation, Soil Moisture'"
        )
        custom_features = [feature.strip() for feature in custom_features.split(",") if feature.strip()]
        
        # Combine predefined and custom features
        features = predefined_features + custom_features
        
        # Sample size selection
        n_samples = st.sidebar.slider(
            "Number of Samples", 
            min_value=100, 
            max_value=10000, 
            value=1000
        )
        
        # Noise level
        noise_level = st.sidebar.slider(
            "Noise Level", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.01
        )
        
        # Generate data button
        if st.sidebar.button("Generate Data"):
            # Synthetic data generation
            data = self.generate_synthetic_data(
                features, 
                n_samples, 
                noise_level
            )
            
            # Store in session state
            st.session_state.generated_data = data
            
            # Display generated data
            st.subheader("Generated Synthetic Data")
            st.dataframe(data)
        
        # Display existing data if available
        if st.session_state.generated_data is not None:
            st.subheader("Current Dataset")
            st.dataframe(st.session_state.generated_data)
    
    def generate_synthetic_data(self, features, n_samples, noise_level):
        """
        Generate synthetic data with specified features and realistic correlations
        """
        np.random.seed(42)
    
    # Create base data
        data = pd.DataFrame()
    
        # Generate predefined features
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
                    data["Pressure"] = standard_pressure * np.exp(-0.0001 * data["Altitude"])
                    data["Pressure"] += np.random.normal(0, 20, n_samples)
                else:
                    data["Pressure"] = np.random.normal(1000, 50, n_samples)
            elif feature == "Humidity":
                if "Temperature" in data.columns:
                    data["Humidity"] = np.clip(100 - (data["Temperature"] * 2) + 
                                        np.random.normal(50, 10, n_samples), 0, 100)
                else:
                    data["Humidity"] = np.clip(np.random.normal(60, 15, n_samples), 0, 100)
            elif feature == "Wind Speed":
                base_wind = np.abs(np.random.normal(5, 2, n_samples))
                if "Altitude" in data.columns:
                    altitude_factor = 0.005 * data["Altitude"]
                    base_wind += altitude_factor
                data["Wind Speed"] = np.clip(base_wind, 0, 30)
            else:
                # Generate random data for custom features
                data[feature] = np.random.normal(0, 1, n_samples)
        
        # Create target variable with realistic relationships
        if len(features) > 0:
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
        Perform Exploratory Data Analysis
        """
        st.header("📊 Exploratory Data Analysis")
        
        # Check if data is generated
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        
        data = st.session_state.generated_data
        
        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())
        
        # Correlation analysis
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)
        
        # Distribution plots
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
        Modeling and simulation section
        """
        st.header("🤖 Modeling and Simulation")
        
        # Check if data is generated
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
        
        data = st.session_state.generated_data
        
        # Prepare data for modeling
        X = data.drop('Target', axis=1)
        y = data['Target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test_scaled)
        
        # Model evaluation
        st.subheader("Model Performance")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance in Predictive Model')
        st.pyplot(plt)
    
    def simulation_page(self):
        """
        Enhanced simulation and scenario analysis page with interactive features
        and detailed results visualization
        """
        st.header("🔮 Scenario Simulation")
    
        # Check if data is generated
        if st.session_state.generated_data is None:
            st.warning("Please generate data first in the Data Generation page.")
            return
    
        data = st.session_state.generated_data
    
        # Prepare data
        X = data.drop('Target', axis=1)
        y = data['Target']
    
        # Train model with cross-validation
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 
                                min_value=10, 
                                max_value=40, 
                                value=20, 
                                help="Percentage of data used for testing")
        
        with col2:
            n_estimators = st.slider("Number of Trees", 
                                   min_value=50, 
                                   max_value=500, 
                                   value=100, 
                                   step=50,
                                   help="Number of trees in Random Forest")
    
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size/100, 
            random_state=42
        )
    
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
    
        # Scenario simulation section
        st.subheader("What-If Scenario Analysis")
    
        # Create tabs for different simulation modes
        tab1, tab2 = st.tabs(["Single Scenario", "Comparison"])
    
        with tab1:
            # Single scenario analysis
            st.markdown("Adjust individual feature values to see their impact on the target variable.")
        
            # Base scenario
            scenario_data = pd.DataFrame(columns=X.columns)
            scenario_data.loc[0] = X.mean()
        
            # Feature selection and adjustment
            selected_features = st.multiselect(
                "Select features to modify",
                X.columns.tolist(),
                default=[X.columns[0]]
            )
        
            # Create sliders for selected features
            for feature in selected_features:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())
            
                scenario_data.loc[0, feature] = st.slider(
                    f"{feature} ({X[feature].name})",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}"
                )
        
            # Make prediction
            scenario_scaled = scaler.transform(scenario_data)
            prediction = rf_model.predict(scenario_scaled)[0]
        
            # Display results
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Value", f"{prediction:.2f}")
            with col2:
                baseline = rf_model.predict(scaler.transform(pd.DataFrame([X.mean()])))[0]
                st.metric("Baseline Prediction", f"{baseline:.2f}")
            with col3:
                change = ((prediction - baseline) / baseline) * 100
                st.metric("% Change from Baseline", f"{change:+.1f}%")
            
        with tab2:
            # Comparison analysis
            st.markdown("Compare multiple scenarios side by side.")
        
            n_scenarios = st.number_input(
                "Number of scenarios to compare",
                min_value=2,
                max_value=4,
                value=2
            )
        
            scenarios = []
            predictions = []
        
            # Create multiple scenarios
            cols = st.columns(n_scenarios)
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"**Scenario {i+1}**")
                    scenario = X.mean().copy()
                
                    # Allow adjusting each feature
                    for feature in X.columns:
                        min_val = float(X[feature].min())
                        max_val = float(X[feature].max())
                        mean_val = float(X[feature].mean())
                    
                        scenario[feature] = st.slider(
                            f"{feature} (S{i+1})",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            key=f"scenario_{i}_{feature}"
                        )
                
                    scenarios.append(scenario)
                    # Make prediction for this scenario
                    pred = rf_model.predict(scaler.transform([scenario]))[0]
                    predictions.append(pred)
        
            # Display comparison results
            st.subheader("Scenario Comparison")
            comparison_df = pd.DataFrame({
                f"Scenario {i+1}": [pred] for i, pred in enumerate(predictions)
            }, index=['Predicted Value'])
            st.dataframe(comparison_df.style.highlight_max(axis=1))
        
            # Visualization of scenario comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            scenarios_df = pd.DataFrame(scenarios, columns=X.columns)
            scenarios_df = (scenarios_df - scenarios_df.mean()) / scenarios_df.std()
            scenarios_df.index = [f"Scenario {i+1}" for i in range(n_scenarios)]
            sns.heatmap(scenarios_df, center=0, cmap='coolwarm', annot=True, fmt='.2f')
            plt.title('Feature Values Comparison (Standardized)')
            st.pyplot(fig)

        # Add feature importance reminder
        st.info("""
        💡 **Tip**: Remember to consider feature importance when adjusting values. 
        Features with higher importance will have a stronger impact on the prediction.
        """)
    
    def conclusion_page(self):
        """
        Project conclusion and key takeaways
        """
        st.header("🏁 Project Conclusion")
        
        st.markdown("""
        ### Key Learnings from Modeling and Simulation Project
        
        1. **Data Generation**: Creating synthetic data with controllable properties
        2. **Exploratory Analysis**: Understanding data characteristics through visualization
        3. **Modeling**: Applying machine learning techniques for prediction
        4. **Simulation**: Exploring scenarios and understanding model behavior
        
        ### Next Steps and Recommendations
        - Experiment with different data generation techniques
        - Try various machine learning algorithms
        - Explore more complex modeling scenarios
        - Apply these techniques to real-world datasets
        """)
        
        st.info("""
        💡 **Continuous Learning**: 
        Modeling and simulation are powerful techniques that require practice and 
        continuous exploration. Keep experimenting and learning!
        """)
    
    def main(self):
        """
        Main application workflow
        """
        # Sidebar navigation
        pages = {
            "🏠 Introduction": self.introduction_page,
            "🔢 Data Generation": self.data_generation_page,
            "📊 Exploratory Analysis": self.exploratory_analysis_page,
            "🤖 Modeling": self.modeling_page,
            "🔮 Simulation": self.simulation_page,
            "🏁 Conclusion": self.conclusion_page
        }
        
        # Page selection
        st.sidebar.title("Navigation")
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Run selected page
        pages[selection]()

# Run the application
if __name__ == "__main__":
    app = ModelingSimulationApp()
    app.main()