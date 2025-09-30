import pandas as pd
import pickle
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

model = None  # global model variable

# 1️⃣ TRAIN ENDPOINT
@app.route('/train', methods=['POST'])
def train_model():
    global model
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Encode categorical columns (case-insensitive)
        df['sex_enc'] = df['sex'].apply(lambda x: 1 if x.lower() == 'female' else 0)
        df['smoker_enc'] = df['smoker'].map({'yes': 1, 'no': 0})

        X = df[['age', 'sex_enc', 'bmi', 'children', 'smoker_enc']]
        y = df['charges']

        model = LinearRegression()
        model.fit(X, y)

        # Save model
        pickle.dump(model, open("model.pkl", "wb"))

        return jsonify({"message": "OK"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2️⃣ TEST ENDPOINT
@app.route('/test', methods=['POST'])
def test_model():
    global model
    try:
        if model is None:
            try:
                model = pickle.load(open("model.pkl", "rb"))
            except FileNotFoundError:
                return jsonify({"error": "Model not trained yet"}), 400

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        # Encode categorical columns (case-insensitive)
        df['sex_enc'] = df['sex'].apply(lambda x: 1 if x.lower() == 'female' else 0)
        df['smoker_enc'] = df['smoker'].map({'yes': 1, 'no': 0})

        X = df[['age', 'sex_enc', 'bmi', 'children', 'smoker_enc']]
        y_true = df['charges']

        y_pred = model.predict(X)
        mse = mean_squared_error(y_true, y_pred)

        return jsonify({"MeanSquaredError": round(mse, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3️⃣ PREDICTION ENDPOINT
@app.route('/prediction', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            try:
                model = pickle.load(open("model.pkl", "rb"))
            except FileNotFoundError:
                return jsonify({"error": "Model not trained yet"}), 400

        data = request.get_json(force=True)
        required_keys = ["age", "sex", "bmi", "children", "smoker"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        age = data["age"]
        sex = 1 if data["sex"].lower() == "female" else 0
        bmi = data["bmi"]
        children = data["children"]
        smoker = 1 if data["smoker"].lower() == "yes" else 0

        X_test = [[age, sex, bmi, children, smoker]]
        prediction = round(max(500, model.predict(X_test)[0]), 2)

        return jsonify({"Your Premium is": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Only required change: fix debug reloader to avoid 404
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
