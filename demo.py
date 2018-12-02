import cv2
import numpy as np
from PIL import Image
from keras import models

model = models.load_model('model_leNet.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')

        im = im.resize((28,28))
        img_array = np.array(im)

        
        img_array = np.expand_dims(img_array, axis=0)

        prediction = int(model.predict(img_array)[0][0])

        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()