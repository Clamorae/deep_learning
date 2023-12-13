#Import Necessary Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


#Load the MNIST dataset
#(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = X_train.reshape((X_train.shape[0], 64, 64))
x_test = X_test.reshape((X_test.shape[0], 64, 64))

#Normalize the data to [0, 1] range
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#Flatten the images as the autoencoder will be fully connected
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoding_dim = 32 # Size of the encoded representations

#Define the encoder
input_img = tf.keras.Input(shape=(4096,))
encoded = layers.Dense(encoding_dim, activation= 'relu') (input_img)

#Define the decoder
decoded = layers.Dense (4096, activation='sigmoid') (encoded)

#Combine the encoder and the decoder into an autoencoder model
autoencoder = models. Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

#Train the model
history = autoencoder.fit(x_train, x_train,
                          epochs=50,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

#otain the reconstructed images
decoded_imgs = autoencoder.predict(x_test)

#Plot original and reconstructed images
n=10
for i in range(n):
  #Display original
  ax = plt.subplot(2, n, i+1)
  plt.imshow(x_test[i].reshape(64, 64))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible (False)

  #Display reconstruction
  ax = plt.subplot(2, n, i+1+n)
  plt.imshow(decoded_imgs[i].reshape(64, 64))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)