import os
import io
import time
import utils

from PIL import Image
from model import SiameseModel
from flask import Flask, request, jsonify

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

app = Flask(__name__)

cur_dir = os.getcwd()
model_path = os.path.join(cur_dir, "model")


@app.route('/')
def status():
    return "This is a Rest API that manages the product recognition system, developed for the Emptio application."


@app.route('/predict', methods=["POST"])
def predictRoute():
    body = request.json

    if not "image" in body.keys():
        return "missing param ['image']", 400

    image = utils.decodeBase64(body["image"])
    if not image:
        return "invalid param ['image'] ::: expected base64 data", 400

    image = Image.open(io.BytesIO(image))
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.ADAPTIVE)
    tmp_name = "temp_{}.jpg".format(time.time())
    tmp_path = os.path.join(cur_dir, tmp_name)
    image.save(tmp_path)

    model = SiameseModel(model_path, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    prediction = model.predict(tmp_path)
    os.remove(tmp_path)

    if not prediction:
        return "failed to predict", 500

    return jsonify({"prediction": prediction})


@app.route('/models', methods=["POST"])
def addModelRoute():
    body = request.json

    if not "image" in body.keys():
        return "missing param ['image']", 400

    if not "class_name" in body.keys():
        return "missing param ['class_name']", 400

    image = utils.decodeBase64(body["image"])
    if not image:
        return "invalid param ['image'] ::: expected base64 data", 400

    # TODO - check if class_name already exists

    # TODO - Pre process image

    # TODO - save new image in /models with class_name name

    return jsonify({"developing": True})


@app.route('/models/<class_name>', methods=["DELETE"])
def deleteModelRoute(class_name):
    if (not class_name) or class_name == "":
        return "invalid param ['class_name'] ::: expected string"

    # TODO - Validate if class exists

    # TODO - delete model from /models

    return jsonify({"developing": True})


@app.errorhandler(404)
def notFoundRoute(error):
    return "Route not found! Please check your URL.", 404


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
