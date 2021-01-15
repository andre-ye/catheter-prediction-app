from flask import Flask, render_template, sessions, url_for, redirect, request
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, FileField
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
FILENAME = None


def return_predictions(img):
    print(os.listdir())
    # img_path = f"./photos/{FILENAME}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save("static/media/photo")
    return redirect(url_for('prediction'))


@app.route("/prediction")
def prediction():
    results = "[1, 2, 3, 4, 5, 6]"
    return render_template("prediction.html", results=results)


if __name__ == '__main__':
    app.run()
