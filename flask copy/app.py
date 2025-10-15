from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load all models and class names
# DenseNet201
densenet_model = load_model("../models/DenseNet201_model80ft.h5")
densenet_classes = sorted(os.listdir("../new_dataset/80-20/train"))
# Xception
xception_model = load_model("../models/Xception_model80ft.h5")
xception_classes = densenet_classes  # If both use the same classes
# InceptionV3
inception_model = load_model("../models/inceptionv3_final_80_20 (1).h5")
inception_classes = sorted(os.listdir("../new_dataset/80-20/train"))
# ResNet50V2
resnet_model = load_model("../models/resnet50V2_final80_20 (3).h5")
resnet_classes = sorted(os.listdir("../new_dataset/80-20/train"))

def predict_single_image(img_path, model, class_labels, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    file_path = None
    selected_model = "densenet"
    if request.method == "POST":
        file = request.files.get("file")
        selected_model = request.form.get("model", "densenet")
        if file and file.filename:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            if selected_model == "xception":
                prediction = predict_single_image(file_path, xception_model, xception_classes, (299, 299))
            elif selected_model == "inception":
                prediction = predict_single_image(file_path, inception_model, inception_classes, (299, 299))
            elif selected_model == "resnet":
                prediction = predict_single_image(file_path, resnet_model, resnet_classes, (250, 250))
            else:
                prediction = predict_single_image(file_path, densenet_model, densenet_classes, (250, 250))
    return render_template("index.html", prediction=prediction, file_path=file_path, selected_model=selected_model)