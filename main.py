import cv2
import numpy as np
from keras.models import load_model

np.set_printoptions(suppress=True)
model = load_model("head.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

window_width = 800
window_height = 600

while True:
    ret, image = camera.read()
    image_display = cv2.resize(image, (window_width, window_height))

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image, verbose=None)
    index = np.argmax(prediction)
    class_name = class_names[index]

    is_head_detected = class_name[2:].strip().lower() == "head"
    window_title = "Head Detected" if is_head_detected else "No Head Detected"
    
    cv2.imshow("Webcam Image", image_display)
    cv2.setWindowTitle("Webcam Image", window_title)

    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
