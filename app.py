import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# starting point of the application
app = Flask(__name__)
# Load the model
regmodel = pickle.load(file=open("regmodel.pkl", mode="rb"))

@app.route("/") # go to the home page (home path)
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"]) # this is a POST request
def predict_api():
    data=request.json["date"]
    print(data)

# once we get the data we need to do the standardization
