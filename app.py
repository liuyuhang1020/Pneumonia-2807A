from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from keras.models import load_model
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save("./static/" + filename)
        model = load_model("pneumonia_model")
        img = Image.open("./static/" + filename)
        img = img.resize((150, 150))
        img = np.asarray(img, dtype="float32")
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        pred = model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))

if __name__ == "__main__":
    app.run()