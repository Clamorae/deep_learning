import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers,models
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

#SECTION - Constant
# --------------------------------- CONSTANT --------------------------------- #

encoding_dim = 32
BATCH_SIZE = 32
EPOCHS  = 25

#NOTE - 320*64*64 / 80*64*64
#SECTION - Retrieve Data
# ------------------------------- RETRIEVE DATA ------------------------------ #
data = fetch_olivetti_faces()
faces = data.images
train, test = train_test_split(faces,test_size=0.2)

train = train.reshape(len(train),np.prod(train.shape[1:]))
test = test.reshape(len(test),np.prod(test.shape[1:]))

#SECTION - Model creation
# ------------------------------ MODEL CREATION ------------------------------ #
input_img = keras.Input(shape=(4096,))
encoded = layers.Dense(encoding_dim, activation="relu")(input_img)
decoded = layers.Dense(4096,activation="sigmoid")(encoded)
autoencoder = models.Model(input_img, decoded)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

history = autoencoder.fit(train,train,batch_size=BATCH_SIZE,epochs=EPOCHS,shuffle=True,validation_data=(test,test))

decoded_imgs = autoencoder.predict(test)

for i in range(EPOCHS):
    ax = plt.subplot(2, EPOCHS, i+1)
    plt.imshow(test[i].reshape(64,64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, EPOCHS, i+1)
    plt.imshow(decoded_imgs[i].reshape(64,64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig("./face.jpg")