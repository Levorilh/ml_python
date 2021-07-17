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
from mlp import load_mlp_model, predict_mlp_model_classification, destroy_mlp_model, create_mlp_model
from linear import load_linear_model, predict_linear_model_classif, destroy_linear_model
import numpy as np

app = Flask(__name__)

MONUMENTS = ["Arc de Triomphe", "Hotel de ville", "Jardin des tuileries", "Moulin Rouge", "Musee d'Orsay",
             "Palais de l'Elysée", "Place de la concorde", "pont-neuf", "Inconnu"]

MODELS_FILENAMES = {
    'mlp': [],
    'lin': [],
}
MODEL_NAMES = ["PMC", "Linéaires"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_URL = "static/images/"
app.config["UPLOAD_URL"] = UPLOAD_URL
UPLOAD_ROOT = os.path.join(BASE_DIR, "static/images")
app.config["UPLOAD_ROOT"] = UPLOAD_ROOT

p_model_curr, _ = create_mlp_model([192, 5, 8])
dir_models = os.getcwd() + "/../models/"


@app.route("/")
def index():
    loadModels(MODELS_FILENAMES)
    print(MODELS_FILENAMES)
    return render_template("index.twig", image="https://www.francetourisme.fr/images/musees_expositions.jpg",
                           modelFiles=MODELS_FILENAMES, modelNames=MODEL_NAMES)


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
    # print(a_fl)
    # print(len(a_fl))

    if session.get('modeltype') == "Linéaires":
        pred = predict_linear_model_classif(p_model_curr, session.get("modelSize"), a_fl)
    else:
        pred = predict_mlp_model_classification(p_model_curr, a_fl, len(MONUMENTS) - 1)

    print(pred)

    return render_template("index.twig", image=image, modelFiles=MODELS_FILENAMES, modelNames=MODEL_NAMES)


@app.route("/setModel", methods=["POST"])
def setModel_route():
    session['modeltype'] = request.form["type"]

    # if session['modeltype'] == "Linéaires":
    #     # session['modelSize'], session['model'] = load_linear_model(curr_dir / 'lin' / request.form["file"])
    #     # session['modelSize'], p_model_curr = load_linear_model(dir_models + "lin/" + request.form["file"])
    # else:
        # p_model_curr = load_mlp_model(dir_models + "mlp/" + request.form["file"])

    print(p_model_curr)

    return "", 201


if __name__ == '__main__':
    app.secret_key = "3A-IABD2"
    app.jinja_env.filters['zip'] = zip

    app.run(debug=True)
