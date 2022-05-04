import pickle
import os
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # we can return content and status code
    return "<h1>Welcome to our flask app!!</h1>", 200

# now for the /predict endpoint
@app.route("/predict", methods=["GET"])
def predict():
    # parse the query string to get our
    # instance attribute values from the client
    fixed_acidity = request.args.get("fixed acidity", "")
    citric_acid = request.args.get("citric acid", "")
    residual_sugar = request.args.get("residual sugar", "")
    pH = request.args.get("pH", "")
    sulphates = request.args.get("sulphates", "")
    alcohol = request.args.get("alcohol", "")
    print([fixed_acidity, citric_acid, residual_sugar, pH, sulphates, alcohol])

    prediction = predict_wine_quality([[fixed_acidity, citric_acid, residual_sugar, pH, sulphates, alcohol]])

    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def bin_values(input_instance, classifier, bin_size=10):
    binned_instance = []
    for val_idx in range(len(input_instance)):
        X_col = [row[val_idx] for row in classifier.X]
        minv = min(X_col)
        maxv = max(X_col)
        val = input_instance[val_idx]
        b = int((val-minv) / (maxv - minv) * bin_size)
        binned_instance.append(b)
    return binned_instance

def predict_wine_quality(instance):
    infile = open("tree.p", "rb")
    wine_random_forest = pickle.load(infile)
    infile.close()

    try:
        return wine_random_forest.predict(instance, bin=True)
    except:
        print("Error")
        return None


if __name__ == "__main__":
    port = os.environ.get("PORT", 5002)
    app.run(debug=False, port=port, host="0.0.0.0")
