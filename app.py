from flask import Flask, request, render_template_string
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("flight_fare_model.pkl", "rb"))

HTML_PAGE = open("index.html").read()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        airline = int(request.form["airline"])
        stops = int(request.form["stops"])
        duration = int(request.form["duration"])
        day = int(request.form["day"])
        month = int(request.form["month"])

        features = np.array([[airline, stops, duration, day, month]])
        prediction = round(model.predict(features)[0], 2)

    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
