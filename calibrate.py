import numpy as np
import cv2
import tensorflow as tf
import os

model = tf.keras.models.load_model("model/anomaly_model.h5",compile=False)

data_path = "frames/train"

errors = []

for img in os.listdir(data_path):

    path = os.path.join(data_path,img)

    img = cv2.imread(path,0)
    img = cv2.resize(img,(128,128))
    img = img/255.0
    img = img.reshape(1,128,128,1)

    recon = model.predict(img,verbose=0)

    error = np.mean((img-recon)**2)

    errors.append(error)

mean = np.mean(errors)
std = np.std(errors)

print("MEAN:",mean)
print("STD:",std)

np.save("model/mean.npy",mean)
np.save("model/std.npy",std)
