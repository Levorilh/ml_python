import datetime
import os
from pathlib import Path

from PIL import Image

from flask import Flask, render_template, session, request, redirect
import random
from PIL import Image

from numpy import shape
from werkzeug.utils import secure_filename

from loadModels import loadModels
from mlp import load_mlp_model, predict_mlp_model_classification, destroy_mlp_model
from linear import load_linear_model, predict_linear_model_classif, destroy_linear_model
import numpy as np

app = Flask(__name__)

MONUMENTS = ["Arc de Triomphe", "Hotel de ville", "Jardin des tuileries", "Moulin Rouge", "Musee d'Orsay",
             "Palais de l'Elysée", "Place de la concorde", "pont-neuf", "Inconnu"]
MODELS_FILENAMES = {
    'mlp': [],
    'lin': [],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_URL = "static/images/"
app.config["UPLOAD_URL"] = UPLOAD_URL
UPLOAD_ROOT = os.path.join(BASE_DIR, "static/images")
app.config["UPLOAD_ROOT"] = UPLOAD_ROOT

p_model_curr = None
dir_models = os.getcwd() + "/../models/"


@app.route("/index")
def index():
    return render_template("index.html", image="https://www.francetourisme.fr/images/musees_expositions.jpg",
                           prediction_score=None, labels=None, scores=None, prediction_label=None)


@app.route("/predictImage", methods=["GET", "POST"])
def predictImage():
    f = request.files["filePath"]
    fs = secure_filename(f.filename)
    f.save(os.path.join(app.config["UPLOAD_ROOT"], fs))
    image = str(os.path.join(app.config["UPLOAD_URL"], f.filename))
    img = Image.open(image)
    im_re = img.resize((8, 8))
    a = np.asarray(im_re)
    a_fl = a.flatten() / 255.
    print(a_fl)
    print(len(a_fl))

    return render_template("index.html", image=image)


@app.route("/")
def hello_world(predict=None):
    return render_template('home.twig', modelFiles=MODELS_FILENAMES, modelNames=['PMC', 'Linéaires'], predict=predict)


@app.route("/predict", methods=["GET","POST"])
def predict_route():
    accuracy = 60 + random.random() * 40

    print(request.files)
    image = request.files.get('image')
    print("bonsoirrrrrrrrrrrr ", image)
    img = Image.open(image)
    print("bonjour", img)

    # data = np.array(request.files['file']).flatten()

    # if session['modeltype'] == "Linéaires":
    #     model_prediction = predict_linear_model_classif(p_model_curr, session['modelSize'], data)
    # else:
    #     model_prediction = predict_mlp_model_classification(p_model_curr, data, 8)

    # print(model_prediction)
    className = "oui"
    # TODO call predict with image + p_model from session.
    return hello_world(render_template('prediction.twig', className=className, accuracy=accuracy))


@app.route("/setModel", methods=["POST"])
def setModel_route():
    session['modeltype'] = request.form["type"]

    if request.form["type"] == "Linéaires":
        # session['modelSize'], session['model'] = load_linear_model(curr_dir / 'lin' / request.form["file"])
        session['modelSize'], p_model_curr = load_linear_model(dir_models + "lin/" + request.form["file"])
    else:
        p_model_curr = load_mlp_model(dir_models + "mlp/" + request.form["file"])

    return "", 201


if __name__ == '__main__':
    # loadModels(MODELS_FILENAMES)

    app.secret_key = "3A-IABD2"

    app.jinja_env.filters['zip'] = zip
    app.run(debug=True)
