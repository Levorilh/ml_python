import datetime
import os
from pathlib import Path

from flask import Flask, render_template, session, request
import random
from PIL import Image

import urllib3
urllib3.disable_warnings()

from numpy import shape
from werkzeug.utils import secure_filename

from loadModels import loadModels
from mlp import load_mlp_model, predict_mlp_model_classification, destroy_mlp_model
from linear import load_linear_model, predict_linear_model_classif, destroy_linear_model
import numpy as np

app = Flask(__name__)

MONUMENTS = ['moulin-rouge',
 'palais-de-l-elysee',
 'pont-neuf',
 'place-de-la-concorde',
 'jardin-des-tuileries',
 'hotel-de-ville',
 'arc-de-triomphe',
 'musee-d-orsay']
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
                           prediction_score=50, labels=MONUMENTS, scores=[0, 0, 1, 0, 0, 0, 0, 0], prediction_label="pont-neuf")


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

    model = load_mlp_model("../models/mlp/MLP_50000_8x8_8_t_acc-34.56_v_acc-32.36.txt")
    predictions = predict_mlp_model_classification(model, a_fl, 8)
    print(predictions)
    return render_template("index.html",
                           image=image,
                           prediction_label=MONUMENTS[np.argmax(predictions)],
                           scores=predictions,
                           labels=MONUMENTS,
                           prediction_score=round(predictions[np.argmax(predictions)] * 100, 2)
                           )


@app.route("/")
def hello_world():
    return render_template('home.twig', modelFiles=MODELS_FILENAMES, modelNames=['PMC', 'Linéaires'])


@app.route("/predict", methods=["POST"])
def predict_route():
    accuracy = 60 + random.random() * 40

    print(request.files)
    data = np.array(request.files['file']).flatten()

    # if session['modeltype'] == "Linéaires":
    #     model_prediction = predict_linear_model_classif(p_model_curr, session['modelSize'], data)
    # else:
    #     model_prediction = predict_mlp_model_classification(p_model_curr, data, 8)

    # print(model_prediction)
    className = "oui"
    # TODO call predict with image + p_model from session.
    return render_template('prediction.twig', className=className, accuracy=accuracy)


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
