from flask import Flask, render_template
import random

app = Flask(__name__)

MONUMENTS = ["Tour eiffel", "Arc de Triomphe", "Notre dame", "Inconnu"]


@app.route("/")
def hello_world():
    return render_template('home.twig')


@app.route("/predict", methods=["POST"])
def predict_route():
    className = MONUMENTS[random.randint(0, len(MONUMENTS) - 1)]

    accuracy = 60 + random.random() * 40
    return render_template('prediction.twig', className=className, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
