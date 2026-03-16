from flask import Flask, jsonify
import os

# import your main project function
from main import main

app = Flask(__name__)

@app.route("/")
def home():
    return "Carbon Footprint Tracker / Customer Churn Model is Running"

@app.route("/run-model")
def run_model():
    main()
    return jsonify({"message": "Model executed successfully"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)