import os
from PIL import Image
from flask import Flask, render_template, session, request
from werkzeug.utils import secure_filename
from loadModels import loadModels
from mlp import *
from linear import *
import numpy as np

app = Flask(__name__)

MODEL_NAMES = ["PMC", "Linéaires"]

MODELS_FILENAMES = {
    'mlp': [],
    'lin': [],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_URL = "static/images/"
UPLOAD_ROOT = os.path.join(BASE_DIR, "static/images")

app.config["UPLOAD_URL"] = UPLOAD_URL
app.config["UPLOAD_ROOT"] = UPLOAD_ROOT

dir_models = os.getcwd() + "/../models/"


@app.route("/")
def index():
    for k in MODELS_FILENAMES.keys():
        MODELS_FILENAMES[k] = []

    loadModels(MODELS_FILENAMES)
    print(MODELS_FILENAMES)
    return render_template("index.twig", image="https://www.francetourisme.fr/images/musees_expositions.jpg",
                           modelFiles=MODELS_FILENAMES, modelNames=MODEL_NAMES, scores=None)


@app.route("/predictImage", methods=["POST"])
def predictImage():
    print("bonjour", request)

    f = request.files.get("filePath", None)
    if f is None:
        return "arf"
    fs = secure_filename(f.filename)

    IMG_SIZE = (session.get("img_size1"), session.get("img_size2"))

    f.save(os.path.join(app.config["UPLOAD_ROOT"], fs))
    image = str(os.path.join(app.config["UPLOAD_URL"], f.filename))
    img = Image.open(image)
    im_re = img.resize(IMG_SIZE)
    a = np.asarray(im_re)
    a_fl = a.flatten() / 255.

    if session.get('modeltype') == "Linéaires":
        pred = predict_linear_model_classif(session.get('model'), session.get("modelSize"), a_fl)

        print(pred)

        return render_template("prediction.twig", image=image, modelFiles=MODELS_FILENAMES, modelNames=MODEL_NAMES,
                               labels=" ".join(classes), score=None)

    else:
        pred = predict_mlp_model_classification(session.get('model'), a_fl, 8)
        best_class = np.argmax(pred)
        prediction_label = classes[best_class]
        prediction_score = pred[best_class]

        return render_template("prediction.twig", image=image, modelFiles=MODELS_FILENAMES, modelNames=MODEL_NAMES,
                               scores=pred, prediction_label=prediction_label, labels=classes,
                               prediction_score=round(prediction_score * 100, 2))


@app.route("/setModel", methods=["POST"])
def setModel_route():
    session['modeltype'] = request.form["type"]

    if request.form["type"] == "Linéaires":
        session['modelSize'], session['model'] = load_linear_model(dir_models + "lin/" + request.form["file"])
    else:
        session['model'] = load_mlp_model(dir_models + "mlp/" + request.form["file"])
        fileparts = request.form["file"].split("x")
        session["img_size1"] = int(fileparts[0][-1])
        session["img_size2"] = int(fileparts[1][0])

    print(session)

    return "", 201


if __name__ == '__main__':
    app.secret_key = "3A-IABD2"
    app.jinja_env.filters['zip'] = zip

    PATH = os.path.join("../data_large/")
    TRAIN = os.path.join(PATH, "train")
    classes = os.listdir(TRAIN)

    app.run(debug=True)
