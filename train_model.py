import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# Path
data_path = "frames/train"

# Load images
data = []

for img in os.listdir(data_path):
    path = os.path.join(data_path, img)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128,128))
    image = image / 255.0

    data.append(image)

data = np.array(data)
data = np.reshape(data, (-1,128,128,1))

print("Total images:", data.shape)

# AutoEncoder Model
input_img = Input(shape=(128,128,1))

x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)

x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer=Adam(), loss='mse')

autoencoder.summary()

# Train
autoencoder.fit(
    data, data,
    epochs=20,
    batch_size=32,
    shuffle=True
)

# Save model
autoencoder.save("model/anomaly_model.h5")

# Second reconstruction model (same model reuse)
autoencoder.save("model/anomaly_model_2.h5")


print("Model trained & saved!")
