import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# starting point of the application
app = Flask(__name__)
# Load the model
regmodel = pickle.load(file=open("regmodel.pkl", mode="rb"))
# Load scaler
scaler = pickle.load(file=open("scaling.pkl", mode="rb"))

@app.route("/") # go to the home page (home path)
def home():
    # every time we go to the home page, it should redirect us to home.html
    # The render_template() in flask will look at a template folder. We need to create it
    # and add a home.hmlt file!
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"]) # this is a POST request
def predict_api():
    data=request.json["data"]
    print(data)
    # Data will come in a dictionary format from the json.
    # Then we take it, convert it to a list, and use np.array to be able to do the reshape,
    # key step to get a 2D data point instead of just 1D (that wouldn't work). Exactly what
    # we did on the Jupyter Notebook
    # print(np.array(list(data.values())).reshape(1,-1))
    # Apply transformation from scaler
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    # Use regression model to predict new value
    output = regmodel.predict(new_data)
    print(output) # to see the output in the cmd
    return jsonify(output[0])

# What I've done:
    # I've activated the venv of this project.
    # Write python app.py on the cmd
    # Accessed the http address that appears in cmd
    # From there I can see the home.html file we created with a simple Hello World!!
    # When trying to access http://127.0.0.1:5000/predict_api, nothing happens. Why?
    # The method is not allowed, because the method is POST. We need to give some information
    # some POST from the client head. That information is the data we want to predict from.
    # Now we need to use Postman to simulate the input of data (POST). 
    # There I created a POST with the URL http://127.0.0.1:5000/predict_api.
    # Introduced the data in a JSON format as follows in raw section:
    # {
#     "data": {
#         "CRIM": 0.00632,
#         "ZN": 18.0,
#         "INDUS": 2.31,
#         "CHAS": 0.0,
#         "NOX": 0.538,
#         "RM": 6.575,
#         "AGE": 65.2,
#         "DIS": 4.090,
#         "RAD": 1.0,
#         "TAX": 296,
#         "PTRATIO": 15.3,
#         "B": 396.90,
#         "LSTAT": 4.98
#     }
# }
    # Make sure you're code is correct, otherwise you'll get plenty of errors from the POST action.
    # I had an error writing the reshape part...

# Now we need to deploy this!
# Instead of creating the prediction in the form of an API, why not create a small web application

# Let's create the html page that allows us to input the data from there
@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)

    output = regmodel.predict(final_input)[0]
    print(output)
    # the return will be a rendered template with the prediction text we want
    return render_template("home.html",prediction_text = "The House price prediction is {}".format(output))
# in order to run this
if __name__ == "__main__":
    app.run(debug = True)
