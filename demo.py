import cv2
import numpy as np
from keras import models

model = models.load_model('model_server.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        im = np.resize(im,(28,28))
        im = np.reshape(im, (28,28,1))
        img_array = np.array(im)

        img_array = np.expand_dims(img_array, axis=0)

        predictions = list(model.predict(img_array)[0])

        maxValue = max(predictions)

        (h, w, l) = frame.shape
        frame = cv2.resize(frame, (int(w * 0.3), int(h * 0.3)))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "{}".format(predictions.index(maxValue)), (30, 100), font, 4,(255,0,0),2,cv2.LINE_AA)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break

cv2.destroyAllWindows()
video.release()