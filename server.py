from flask import Flask, render_template, request
from keras.models import load_model
import os
import cv2
import config

import numpy as np

#run Flask
app = Flask(__name__)

model_vgg16 = load_model(os.path.join("C:/Users/ardra/Downloads/PestClassification-main/models/ICT-w-max.hdf5"))
model_resnet50 = load_model(os.path.join("C:/Users/ardra/Downloads/PestClassification-main/models/ResNet50-w-max.hdf5"))
model_inceptionv3 = load_model(os.path.join("C:/Users/ardra/Downloads/PestClassification-main/models/ICT-w-max.hdf5"))

import pickle
file = open('C:/Users/ardra/Downloads/PestClassification-main/models/le.pkl','rb')
le = pickle.load(file)
file.close()

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        try:
            # get file
            image = request.files['file']
            if image:
                #Save file
                path_to_save = os.path.join("static/file",image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                #Read file, resize the size of the input model
                frame = cv2.imread(path_to_save)
                frame = cv2.resize(frame, dsize = (config.image_size, config.image_size))
                #convert tensor bar
                frame = np.expand_dims(frame, axis=0)

                #run model 
                if request.form.get("models") == "VGG16":
                    predict = model_vgg16.predict(frame)
                    print("VGG16")
                elif request.form.get("models") == "Resnet50":
                    predict = model_resnet50.predict(frame)
                    print("Resnet50")
                else:
                    predict = model_inceptionv3.predict(frame)
                    print("Inception v3")

                list_pred = np.argsort(predict)
                more = "{} : {:.2f}%; {} : {:.2f}%; {} : {:.2f}%".format(le.inverse_transform([np.argmax(predict)])[0],predict[0][list_pred[0][-1]]*100,le.inverse_transform([list_pred[0][-2]])[0],predict[0][list_pred[0][-2]]*100,le.inverse_transform([list_pred[0][-3]])[0],predict[0][list_pred[0][-3]]*100)
                predict_name = le.inverse_transform([np.argmax(predict)])[0]

                return render_template("index.html", 
                        image = image.filename, 
                        msg="Upload Successfull", 
                        models=request.form.get("models"),
                        predict_name=predict_name,
                        more=more)
            else:
                return render_template('index.html', msg='Please select the file to upload')
        except Exception as ex:
            print(ex)
            return render_template('index.html', msg="Image not recognized!")
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False)