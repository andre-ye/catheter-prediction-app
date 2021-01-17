from flask import Flask, render_template, sessions, url_for, redirect, request, abort
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, FileField
from werkzeug.utils import secure_filename
import os

import numpy as np
import pandas as pd

## Keras and Tf ##

import tensorflow as tf
import tensorflow.keras.callbacks as C
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.optimizers as O
import tensorflow.keras.utils as U
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import efficientnet.tfkeras as efn


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
FILE_DIR = "static/media/photo.png"
IMG_SIZES = (512, 512)
base_model = tf.keras.applications.EfficientNetB5


def build_model(dim=331, base_model=base_model, n_labels=11, ):
    inp = L.Input(shape=(dim, dim, 3))
    base = base_model(input_shape=(dim, dim, 3), weights=None, include_top=False)
    x = base(inp)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(n_labels, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=x)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = "binary_crossentropy"
    model.compile(optimizer=opt, loss=loss, metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)])
    return model


def return_predictions():

    file_bytes = tf.io.read_file(FILE_DIR)
    img = tf.image.decode_png(file_bytes, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, IMG_SIZES).numpy()
    img = np.expand_dims(img, axis=0)

    # dset = dset.map(decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    model = build_model(dim=IMG_SIZES[0])
    model.load_weights(f"static/model_weights/RANZCRModelFold0.h5")
    predictions = model.predict(img, verbose=1)

    return predictions


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
    return render_template("prediction.html", results=results)


if __name__ == '__main__':
    app.run()
