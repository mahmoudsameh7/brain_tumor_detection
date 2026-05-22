import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


model = load_model("braintumor10Epoccategorical.h5")


image = cv2.imread(r"C:\Users\hp\OneDrive\Documents\brain_test\test1\Te-me_0015.jpg")
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)


img = img / 255.0

input_img = np.expand_dims(img, axis=0)


prediction = model.predict(input_img)
print("Raw prediction probabilities:", prediction)
result = np.argmax(prediction, axis=1)
print("Prediction result:", result)


if result[0] == 0:
    print("Predicted: NO TUMOR")
else:
    print("Predicted: TUMOR")
