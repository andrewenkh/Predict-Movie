
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify, send_from_directory


# Load dataset
df = pd.read_csv("movie_data.csv")
print("Dataset Columns:", df.columns)
df.columns = df.columns.str.strip()

# Data Cleaning
def clean_currency(value):
    if isinstance(value, str):
        return float(value.replace(',', '').strip())
    return value
df['Net Revenue (MNT)'] = df['Net Revenue (MNT)'].apply(clean_currency)
df['Gross Revenue (MNT)'] = df['Gross Revenue (MNT)'].apply(clean_currency)
df['ATP (Avg. Ticket Price, MNT)'] = df['ATP (Avg. Ticket Price, MNT)'].apply(clean_currency)
df['Admits'] = df['Admits'].apply(clean_currency)
df['Sessions'] = df['Sessions'].astype(float)
df['Occupancy %'] = df['Occupancy %'].str.replace('%', '').astype(float)
column_mapping = {
    'Production Company': 'ProductionCompany',  
    'ATP (Avg. Ticket Price, MNT)': 'ATP'
}
df.rename(columns=column_mapping, inplace=True)
df.dropna(subset=['Gross Revenue (MNT)'], inplace=True)
df = df[df['Sessions'] >= 30]

# Compute average values for numerical features
avg_sessions = df['Sessions'].mean()
avg_admits = df['Admits'].mean()
avg_atp = df['ATP'].mean()
avg_occupancy = df['Occupancy %'].mean()

# Selecting features and target variable
features = ['Genre', 'Director', 'ProductionCompany']
target = 'Gross Revenue (MNT)'

# Handling categorical data
categorical_features = ['Genre', 'Director', 'ProductionCompany']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model selection
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f} MNT')

# ----------------------------
# Flask backend
# ----------------------------

from flask import Flask, request, jsonify, send_from_directory
# ... your other imports and ML code ...

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(".", "ui.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    genre = data.get("genre", "")
    director = data.get("director", "")
    production_company = data.get("productionCompany", "")

    new_movie = pd.DataFrame({
        "Genre": [genre],
        "Director": [director],
        "ProductionCompany": [production_company]
    })

    prediction = pipeline.predict(new_movie)[0]

    return jsonify({
        "title": title,
        "prediction": float(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)


