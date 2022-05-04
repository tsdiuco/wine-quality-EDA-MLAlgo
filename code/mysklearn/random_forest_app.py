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
    residul_sugar = request.args.get("residual+sugar", "")
    alcohol = request.args.get("alcohol", "")
    citric_acid = request.args.get("citric+acid", "")
    fixed_acid = request.args.get("fixed+acid", "")
    print([residul_sugar, alcohol, citric_acid, fixed_acid])

    prediction = predict_wine_quality([residul_sugar, alcohol, citric_acid, fixed_acid])

    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def predict_wine_quality(instance):
    infile = open("tree.p", "rb")
    wine_random_forest = pickle.load(infile)
    infile.close()

    try:
        return wine_random_forest.predict(instance)
    except:
        print("Error")
        return None


if __name__ == "__main__":
    port = os.environ.get("PORT", 5001)
    app.run(debug=False, port=port, host="0.0.0.0")
