[README.md](https://github.com/user-attachments/files/34016488/README.md)
# Used Car Price Predictor

A Streamlit web app that predicts the selling price of a used car from its details. The model is a regression model (Linear, Ridge and Lasso were compared) trained on used car sales data.

## Features

- Simple web form to enter car details
- Instant price prediction
- Categorical inputs handled with saved label encoders
- Model trained on log-transformed price, converted back to the actual price in the app

## Tech Stack

- Python
- Streamlit
- pandas, NumPy
- scikit-learn
- joblib

## Project Structure

```
├── app.py                        # Streamlit app
├── linear_regression_model.pkl   # Trained model
├── encoders.pkl                  # Fitted label encoders for categorical columns
├── feature_order.pkl             # Column order expected by the model
├── requirements.txt              # Dependencies
└── Linear_Ridge_&_Lasso_Regression.ipynb   # Model training notebook
```

## How to Run Locally

1. Clone the repository
   ```
   git clone https://github.com/YOUR_USERNAME/car-price-predictor.git
   cd car-price-predictor
   ```
2. (Optional) Create and activate a virtual environment
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies
   ```
   python -m pip install -r requirements.txt
   ```
4. Start the app
   ```
   python -m streamlit run app.py
   ```
5. Open http://localhost:8501 in your browser.

## How It Works

1. The user enters the car details in the form.
2. Categorical values are converted with the saved encoders.
3. Inputs are arranged in the same column order used during training.
4. The model predicts the log of the price, which the app converts back to the actual price.

## Live Demo

Add your Streamlit link here after deploying: `https://your-app-name.streamlit.app`

## Author

Anushka
