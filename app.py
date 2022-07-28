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
        print("File Received")
        filename = secure_filename(file.filename)
        file.save(app.config("static/"+ filename))
        file = open(app.config("static/" + filename,"r"))
        model = load_model("pneumonia")
        img = Image.open(filename)
        img = img.resize((150,150))
        img = np.asarray(img, dtype="float32")
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img.shape
        pred=model.predict(img)
        return(render_template("index1.html", result=str(pred)))
    else:
        return(render_template("index1.html", result="2"))

if __name__ == "__main__":
    app.run()