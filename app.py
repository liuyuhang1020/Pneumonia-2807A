from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image
import numpy as np

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
        if img.ndim < 3:
            img = np.expand_dims(img, axis=2).repeat(3, axis=2)
        img = img.reshape(1, 150, 150, 3)
        pred = model.predict(img)
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))

if __name__ == "__main__":
    app.run()