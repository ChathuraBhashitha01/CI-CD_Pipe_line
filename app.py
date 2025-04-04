from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('bank-marketing.joblib')

# Define expected fields
EXPECTED_FIELDS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome"
]

# Categorical features that need encoding
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]


@app.route('/')
def index():
    return render_template("frontend.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()  # ✅ CORRECT for JSON body

        # Ensure all expected fields are present
        missing_fields = [field for field in EXPECTED_FIELDS if field not in input_data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        # Dynamically create label encoders and encode categorical features
        label_encoders = {}
        for col in CATEGORICAL_FEATURES:
            if col in input_data:
                value = input_data[col]
                if value is not None:
                    # Initialize label encoder for each field dynamically
                    le = LabelEncoder()
                    input_data[col] = le.fit_transform([value])[0]  # Fit and transform the value
                    label_encoders[col] = le
                else:
                    return jsonify({'error': f'No value for categorical field: {col}'}), 400

        # Convert numerical features to float
        for field in input_data:
            if field not in CATEGORICAL_FEATURES:
                value = input_data[field]
                if value is not None:
                    input_data[field] = float(value)
                else:
                    return jsonify({'error': f'No value for numerical field: {field}'}), 400

        # Convert to NumPy array for prediction
        features = np.array([list(input_data.values())])

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80) # to change the port .
