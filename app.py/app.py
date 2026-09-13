import joblib
import pandas as pd
import gradio as gr


# ============================================================
# 1. LOAD THE TRAINED ML PIPELINE
# ============================================================

model = joblib.load("california_housing_model.pkl")


# ============================================================
# 2. PREDICTION FUNCTION
# ============================================================

def predict_house_value(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity
):

    # Create input DataFrame
    data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    # Feature engineering
    data["rooms_per_household"] = (
        data["total_rooms"] / data["households"]
    )

    data["bedrooms_per_room"] = (
        data["total_bedrooms"] / data["total_rooms"]
    )

    data["population_per_household"] = (
        data["population"] / data["households"]
    )

    # Prediction
    prediction = model.predict(data)[0]

    return f"## 💰 Predicted Median House Value\n\n# ${prediction:,.0f}"


# ============================================================
# 3. CREATE THE APPLICATION INTERFACE
# ============================================================

with gr.Blocks(
    title="California House Value Predictor"
) as app:

    gr.Markdown(
        """
        # 🏠 California House Value Predictor

        ### AI-powered median house-value estimation

        Enter the characteristics of a California housing district
        and let our trained **Random Forest** model estimate its
        median house value.
        """
    )

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 📍 Location")

            longitude = gr.Number(
                label="Longitude",
                value=-118.25
            )

            latitude = gr.Number(
                label="Latitude",
                value=34.05
            )

        with gr.Column():
            gr.Markdown("### 🏡 Property")

            housing_median_age = gr.Number(
                label="Housing Median Age",
                value=25
            )

            total_rooms = gr.Number(
                label="Total Rooms",
                value=2500
            )

            total_bedrooms = gr.Number(
                label="Total Bedrooms",
                value=500
            )

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 👨‍👩‍👧‍👦 Population & Households")

            population = gr.Number(
                label="Population",
                value=1200
            )

            households = gr.Number(
                label="Households",
                value=450
            )

        with gr.Column():
            gr.Markdown("### 💵 Income & Location Type")

            median_income = gr.Number(
                label="Median Income",
                value=6.5
            )

            ocean_proximity = gr.Dropdown(
                choices=[
                    "<1H OCEAN",
                    "INLAND",
                    "ISLAND",
                    "NEAR BAY",
                    "NEAR OCEAN"
                ],
                value="NEAR OCEAN",
                label="Ocean Proximity"
            )

    predict_button = gr.Button(
        "🔮 Predict House Value",
        variant="primary"
    )

    prediction_output = gr.Markdown(
        "### 💰 Prediction will appear here"
    )

    gr.Markdown(
        """
        ---
        **Model:** Random Forest Regression

        **R²:** 0.8067

        *The prediction is an ML estimate based on patterns learned
        from the California Housing dataset. It is not a guaranteed
        market valuation.*
        """
    )

    # Connect button → prediction function
    predict_button.click(
        fn=predict_house_value,
        inputs=[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity
        ],
        outputs=prediction_output
    )


# ============================================================
# 4. LAUNCH APPLICATION
# ============================================================

app.launch(inbrowser=True)