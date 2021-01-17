from flask import Flask, render_template, sessions, url_for, redirect, request, abort
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, FileField
from werkzeug.utils import secure_filename
import os

import numpy as np


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
FILE_DIR = "static/media/photo.png"

def return_predictions():

    predictions = [0.020837, 0.92817, 0.0019, 0.978, 0.324, 0.023, 0.021, 0.97489, 0.0123, 0.413423, 0.019827384]

    return [round(p, 2) for p in predictions]


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return redirect(url_for("upload"))
    uploaded_file.save(f"static/media/photo.png")
    return redirect(url_for('prediction'))


@app.route("/prediction")
def prediction():
    np.set_printoptions(suppress=True)
    results = return_predictions()
    print(results)
    target_cols = ['ETT - Abnormal', 'ETT - Borderline',
                   'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
                   'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
                   'CVC - Borderline', 'CVC - Normal', 'Swan Ganz Catheter Present']
    return render_template("prediction.html", results=results, targets=target_cols)
