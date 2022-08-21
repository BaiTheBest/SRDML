'''
Data generator for CIFAR-MTL
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

T = 10
train_size = 2000
test_size = 10000

X_train_mtl = np.zeros((T*train_size, 32, 32, 3))
y_train_mtl = np.zeros((T*train_size,))

count = np.zeros((T,))
sample_count = 0
for i in range(50000):
    tmp_label = y_train[i,0]
    if count[tmp_label] < train_size:
        X_train_mtl[sample_count,:,:,:] = X_train[i,:,:,:]
        y_train_mtl[sample_count] = tmp_label
        count[tmp_label] += 1
        sample_count += 1
    if (np.min(count)).astype(int) == train_size:
        break

X_train_mtl = tf.image.resize(X_train_mtl, (64,64))
X_test = tf.image.resize(X_test, (64,64))

y_train_mtl = keras.utils.to_categorical(y_train_mtl)
y_test = keras.utils.to_categorical(y_test)
y_train_mtl = y_train_mtl.transpose()
y_test = y_test.transpose()

np.save('X_train', X_train_mtl)
np.save('X_test', X_test)

np.save('y_train', y_train_mtl)
np.save('y_test', y_test)

