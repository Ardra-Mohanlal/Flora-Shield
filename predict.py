import os
import numpy as np
from keras.models import load_model
import cv2
import sys
import config

# Load model da train
my_model = load_model(os.path.join("models","ICT-w-max.hdf5"))
# my_model = load_model(os.path.join("models","VGG16-w-max.hdf5"))
# my_model = load_model(os.path.join("models","ResNet50-w-max.hdf5"))

import pickle
file = open('le.pkl', 'rb')
le = pickle.load(file)
file.close()

# Doc anh tu dong lenh
image_org = cv2.imread(sys.argv[1])

# Resize
image = image_org.copy()
image = cv2.resize(image, dsize=(config.image_size, config.image_size))

# chuyen sang dinh dang tensor
image = np.expand_dims(image, axis=0)

# du doan
predict = my_model.predict(image)
print(predict)
print(np.argsort(predict))
print(np.argsort(predict)[0][-2])
print(le.inverse_transform([np.argsort(predict)[0][-2]]))
print("This picture is: ", le.inverse_transform([np.argmax(predict)]))

# hien thi anh
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2
# viet ket qua
cv2.putText(image_org, le.inverse_transform([np.argmax(predict)])[0], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow("Result",image_org)
cv2.waitKey()
