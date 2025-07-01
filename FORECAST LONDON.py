# 1. IMPORTS
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. DATA LOADING AND PREPROCESSING
def load_and_clean_data(file_path):
    """
    Loads data from a CSV file, cleans column names, and formats temperature columns.
    """
    try:
        #Load the dataset
        df = pd.read_csv(file_path, encoding='latin1')

        #Strip whitespace from column headers
        df.columns = df.columns.str.strip()

        #Rename columns for consistency
        df.rename(columns={
            'ACTUAL THE HIGHEST TEMPERATURE': 'ACTUAL_TEMP',
            'DEEPSEEK_PREDICTION': 'DEEPSEEK_PREDICTION'
        }, inplace=True)

        #Clean all temperature columns to be numeric
        temp_cols = [col for col in df.columns if col not in ['DATES', 'DEEPSEEK_PREDICTION']]
        for col in temp_cols:
            df.loc[:, col] = pd.to_numeric(
                df[col].astype(str).str.extract(r'(\-?\d+\.?\d*)', expand=True)[0],
                errors='coerce'
            )

        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please make sure the path is correct.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


# 3. STATISTICAL ANALYSIS
def analyze_weather_forecasts(df):
    """
    Analyzes weather forecast data from a CSV file to determine the accuracy
    of different forecasting sites.

    The function reads the specified CSV, cleans the data, and then calculates
    various statistics based on the actual column names in the user's file.
    """

    # These are the forecasting sites that will be analyzed, based on your file.
    sites = [col for col in df.columns if col not in ['DATES', 'ACTUAL_TEMP', 'DEEPSEEK_PREDICTION']]

    # Check if all required columns exist after cleaning and renaming.
    for col in sites + ['ACTUAL_TEMP', 'DEEPSEEK_PREDICTION']:
        if col not in df.columns:
            return (f"Error: Required column '{col}' not found in the file.\n"
                    f"Please check your CSV. The columns found were: {df.columns.tolist()}")

    # Clean all temperature columns by extracting only numbers and converting to a numeric type.
    temp_cols = sites + ['ACTUAL_TEMP']
    for col in temp_cols:
        df.loc[:, col] = pd.to_numeric(
            df[col].astype(str).str.extract(r'(\-?\d+\.?\d*)', expand=True)[0],
            errors='coerce'
        )

    # Drop rows where the actual temperature is missing.
    df.dropna(subset=['ACTUAL_TEMP'], inplace=True)

    if df.empty:
        return "No valid data available for analysis after cleaning."

    stats = {}
    for site in sites:
        site_df = df.dropna(subset=[site])
        valid_predictions = len(site_df)

        if valid_predictions > 0:
            diff = site_df[site] - site_df['ACTUAL_TEMP']
            correct_count = (diff == 0).sum()
            over_count = (diff > 0).sum()
            under_count = (diff < 0).sum()

            stats[site] = {
                'correct': correct_count, 'overstated': over_count, 'understated': under_count,
                'correct_%': (correct_count / valid_predictions) * 100,
                'overstated_%': (over_count / valid_predictions) * 100,
                'understated_%': (under_count / valid_predictions) * 100,
                'close_or_correct': (diff.abs() <= 1).sum(),
                'valid_predictions': valid_predictions
            }
        else:
            stats[site] = {
                'correct': 0, 'overstated': 0, 'understated': 0,
                'correct_%': 0, 'overstated_%': 0, 'understated_%': 0,
                'close_or_correct': 0, 'valid_predictions': 0
            }

    stats_df = pd.DataFrame.from_dict(stats, orient='index')

    if not stats_df.empty:
        stats_df['close_or_correct_%'] = (
                    stats_df['close_or_correct'] / stats_df['valid_predictions'].replace(0, np.nan) * 100).fillna(0)
    else:
        stats_df['close_or_correct_%'] = 0

    if not stats_df.empty:
        most_correct_site = stats_df['correct'].idxmax()
        most_incorrect_site = stats_df['correct'].idxmin()
        most_overstated_site = stats_df['overstated'].idxmax()
        most_understated_site = stats_df['understated'].idxmax()
    else:
        most_correct_site, most_incorrect_site, most_overstated_site, most_understated_site = "N/A", "N/A", "N/A", "N/A"

    deepseek_df = df.dropna(subset=['DEEPSEEK_PREDICTION', 'ACTUAL_TEMP'])

    def is_in_range(row):
        prediction_range = str(row['DEEPSEEK_PREDICTION'])
        actual_temp = row['ACTUAL_TEMP']
        try:

            numbers = re.findall(r'(\d+\.?\d*)', prediction_range)

            if len(numbers) < 2:
                return False

            low = float(numbers[0])
            high = float(numbers[1])

            # Ensure low is always the smaller number
            if low > high:
                low, high = high, low

            is_correct = low <= actual_temp <= high

            return is_correct
        except (ValueError, IndexError) as e:
            return False

    if not deepseek_df.empty:
        deepseek_correct_count = deepseek_df.apply(is_in_range, axis=1).sum()
        deepseek_total_predictions = len(deepseek_df)
        deepseek_accuracy = (
                    deepseek_correct_count / deepseek_total_predictions * 100) if deepseek_total_predictions > 0 else 0
    else:
        deepseek_correct_count, deepseek_total_predictions, deepseek_accuracy = 0, 0, 0.0


    summary = []
    summary.append("=" * 60)
    summary.append("           Weather Forecast Accuracy Summary")
    summary.append("=" * 60)
    summary.append(f"\nAnalysis based on {len(df)} days of data from 'EGLC' station.\n")

    summary.append("\n" + "-" * 60)
    summary.append("DEEPSEEK_PREDICTION (Range) Accuracy:")
    summary.append("-" * 60)
    summary.append(f"- Total Predictions Analyzed: {deepseek_total_predictions}")
    summary.append(f"- Correct Guesses (in range): {deepseek_correct_count}")
    summary.append(f"- Accuracy Percentage:        {deepseek_accuracy:.2f}%")

    summary.append("\n\n" + "=" * 80)
    summary.append("Detailed Statistics Table")
    summary.append("=" * 80)

    if not stats_df.empty:
        stats_df.rename(columns={
            'correct': 'Exact', 'overstated': 'Over', 'understated': 'Under',
            'correct_%': 'Exact (%)', 'overstated_%': 'Over (%)', 'understated_%': 'Under (%)',
            'close_or_correct': 'Correct/±1°C', 'valid_predictions': 'Total Guesses',
            'close_or_correct_%': 'Accuracy ±1°C (%)'
        }, inplace=True)

        # Format all percentage columns
        for col in ['Exact (%)', 'Over (%)', 'Under (%)', 'Accuracy ±1°C (%)']:
            stats_df[col] = stats_df[col].map('{:.2f}'.format)

        # Define the order of columns for the final table
        column_order = [
            'Total Guesses', 'Exact', 'Over', 'Under', 'Exact (%)', 'Over (%)', 'Under (%)',
            'Correct/±1°C', 'Accuracy ±1°C (%)'
        ]
        summary.append(stats_df[column_order].to_string())
    else:
        summary.append("No detailed statistics to show.")

    return "\n".join(summary)

# 4. MODEL PERFORMANCE EVALUATION
def evaluate_model_performance(model, X_train, y_train):
    """
    Evaluates the performance of the trained regression model using common metrics.
    """
    # Make predictions on the training data to see how well the model fits
    train_predictions = model.predict(X_train)

    # Calculate performance metrics
    mae = mean_absolute_error(y_train, train_predictions)
    mse = mean_squared_error(y_train, train_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, train_predictions)
    best_alpha = model.alpha_  # Get the best alpha found by RidgeCV

    # --- Create a formatted summary string ---
    summary = [
        "\n" + "=" * 60,
        "         Machine Learning Model Performance Summary",
        "=" * 60,
        f"\nEvaluation of the model's fit on the historical training data:",
        "-" * 60,
        f"- Best Regularization Strength (Alpha): {best_alpha}",
        f"- R-squared (R²):                     {r2:.4f}",
        f"- Mean Absolute Error (MAE):          {mae:.4f}°",
        f"- Root Mean Squared Error (RMSE):     {rmse:.4f}°",
        "-" * 60,
        "\nExplanation of Metrics:",
        "  - R-squared: Closer to 1.0 is better. It represents the percentage",
        "    of the variance in actual temperatures that the model can explain.",
        "  - MAE/RMSE: Closer to 0 is better. These show the average prediction",
        "    error in degrees.",
    ]

    return "\n".join(summary)

# 5. MACHINE LEARNING & PREDICTION
def predict_missing_temperatures(df):
    """
    Trains a Ridge Regression model to predict missing actual temperatures
    and provides a prediction range.
    """

    # Use the cleaned column name 'ACTUAL_TEMP'
    target_col = 'ACTUAL_TEMP'

    # Identify all columns that are forecasts (i.e., not dates or the target)
    forecast_cols = [col for col in df.columns if col not in ['DATES', target_col, 'DEEPSEEK_PREDICTION']]

    # Create a copy to avoid changing the original DataFrame passed to the function
    df_pred = df.copy()

    # Fill missing forecast values with the average of the other forecasts in the same row
    df_pred[forecast_cols] = df_pred[forecast_cols].apply(
        lambda row: row.fillna(row.mean()), axis=1
    )

    # Separate data into a training set (where we have actual temperatures)
    # and a prediction set (where we don't)
    train_df = df_pred[df_pred[target_col].notna()].dropna(subset=forecast_cols, how='all')
    predict_df = df_pred[df_pred[target_col].isna()].dropna(subset=forecast_cols, how='all')

    if train_df.empty:
        print("Model training cannot proceed: No existing temperature data available.")
        return df  # Return original dataframe

    if predict_df.empty:
        print("No missing temperatures to predict.")
        return df  # Return original dataframe

    #Define the original feature DataFrames from the training and prediction sets
    X_train_orig = train_df[forecast_cols].copy()
    X_predict_orig = predict_df[forecast_cols].copy()
    y_train = train_df[target_col]

    #Add the 'forecast_std' feature to both DataFrames
    X_train_orig['forecast_std'] = X_train_orig[forecast_cols].std(axis=1)
    X_predict_orig['forecast_std'] = X_predict_orig[forecast_cols].std(axis=1)

    #Apply PolynomialFeatures to create the final feature sets for the model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train = poly.fit_transform(X_train_orig)
    X_predict = poly.transform(X_predict_orig)

    # Initialize and train the Ridge Regression model
    # Define a list of alpha values you want to test.
    alphas_to_test = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]

    # Initialize RidgeCV. It will test all alphas and find the best one automatically
    # during the .fit() step using cross-validation.
    model = RidgeCV(alphas=alphas_to_test, store_cv_results=True)
    model.fit(X_train, y_train)  # <-- Fit the model FIRST...
    performance_summary = evaluate_model_performance(model, X_train, y_train)
    print(performance_summary)

    # Calculate the model's error margin from the training data
    train_predictions = model.predict(X_train)
    errors = y_train - train_predictions
    error_margin = errors.std()

    # Predict the missing temperatures
    predicted_temps = model.predict(X_predict)

    # Add the predictions and the calculated range to the prediction set
    predict_df[target_col] = predicted_temps
    predict_df['PREDICTED_RANGE'] = [f"{temp - error_margin:.1f} to {temp + error_margin:.1f}" for temp in
                                     predicted_temps]

    # Combine the original training data with the newly predicted data
    final_df = pd.concat([train_df, predict_df]).sort_index()

    return final_df


def predict_future_temperature(df, future_forecasts):
    """
    Trains a model on historical data to predict the temperature for a new,
    single day based on provided forecasts.
    """
    target_col = 'ACTUAL_TEMP'
    train_df = df[df[target_col].notna()].copy()
    forecast_cols = [col for col in df.columns if col not in ['DATES', target_col, 'DEEPSEEK_PREDICTION']]
    train_df[forecast_cols] = train_df[forecast_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
    if train_df.empty:
        return "Model could not be trained: No historical data available."

    # Create the training feature DataFrame and add the standard deviation feature.
    X_train_orig = train_df[forecast_cols].copy()
    X_train_orig['forecast_std'] = X_train_orig[forecast_cols].std(axis=1)
    y_train = train_df[target_col]

    # Create the future prediction DataFrame and add the standard deviation feature.
    future_df_orig = pd.DataFrame([future_forecasts]).reindex(columns=forecast_cols)
    future_df_orig.fillna(future_df_orig.mean(axis=1).iloc[0], inplace=True)
    future_df_orig['forecast_std'] = future_df_orig[forecast_cols].std(axis=1)

    # Apply the PolynomialFeatures transformation to BOTH datasets.
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train = poly.fit_transform(X_train_orig)
    future_df = poly.transform(future_df_orig)  # Use the same fitted transformer

    # Train the Model
    alphas_to_test = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]
    model = RidgeCV(alphas=alphas_to_test).fit(X_train, y_train)

    # Make the Prediction
    # The error margin should be calculated on the transformed training data.
    error_margin = (y_train - model.predict(X_train)).std()
    predicted_temp = model.predict(future_df)[0]

    result = [f"Based on the provided forecasts, the prediction is:", "-" * 50]
    result.append(f"Predicted Highest Temperature: {predicted_temp:.1f}°")
    result.append(
        f"Likely Range:                {predicted_temp - error_margin:.1f}° to {predicted_temp + error_margin:.1f}°")
    result.append("-" * 50)
    return "\n".join(result)

# 6. MAIN EXECUTION
if __name__ == "__main__":
    file_path = r'C:\Users\mkazi\Downloads\POGODA LONDON.csv'

    # Load and clean data ONCE using the new function
    df_cleaned = load_and_clean_data(file_path)

    # Check if data loading was successful before continuing
    if df_cleaned is not None:
        # Pass the cleaned DataFrame to the analysis function
        analysis_summary = analyze_weather_forecasts(df_cleaned.copy()) # Use .copy() to be safe
        print(analysis_summary)

        # Pass the cleaned DataFrame to the prediction function
        df_with_past_predictions = predict_missing_temperatures(df_cleaned.copy())

    #  Forecast the First Available Future Day
    print("\n\n" + "=" * 80)

    # Define the forecast columns first
    forecast_cols = [col for col in df_cleaned.columns if col not in ['DATES', 'ACTUAL_TEMP', 'DEEPSEEK_PREDICTION']]
    future_data = df_cleaned[df_cleaned['ACTUAL_TEMP'].isna()].dropna(subset=forecast_cols, how='all')

    if not future_data.empty:
        # Select the VERY FIRST valid row
        day_to_forecast_row = future_data.iloc[0]
        day_to_forecast_dict = day_to_forecast_row[forecast_cols].to_dict()
        test_date = day_to_forecast_row['DATES']

        print(f"Automatically selected the following day for a new forecast:")
        print(f"  - Date: {test_date}")
        print(f"  - Forecasts Used: {day_to_forecast_dict}")
        print("-" * 80)

        # Run the prediction for this specific day
        future_prediction_summary = predict_future_temperature(df_cleaned, day_to_forecast_dict)
        print(future_prediction_summary)
        print("=" * 80)
    else:
        print("\nCould not find any future days with missing temperatures to forecast.")