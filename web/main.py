from flask import Flask, render_template, session, request
import random

app = Flask(__name__)

MONUMENTS = ["Tour eiffel", "Arc de Triomphe", "Notre dame", "Inconnu"]


@app.route("/")
def hello_world():
    modelIds = {
        "LINEAR": [41, 31, 21, 11],
        "MLP": [4, 3, 2, 1],
    }

    return render_template('home.twig', modelIds=modelIds)


@app.route("/predict", methods=["POST"])
def predict_route():
    className = MONUMENTS[random.randint(0, len(MONUMENTS) - 1)]

    accuracy = 60 + random.random() * 40
    return render_template('prediction.twig', className=className, accuracy=accuracy)


@app.route("/setModel", methods=["POST"])
def setModel_route():
    session['model'] = request.form["id"]

    return "",201


if __name__ == '__main__':
    app.secret_key = "3A-IABD2"
    app.run(debug=True)
