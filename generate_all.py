from cifar10vgg import cifar10vgg
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os

batch_sizes = [32, 64, 128, 256, 512]
learning_rates = [0.01, 0.02, 0.04, 0.08, 0.1]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

for bs in batch_sizes:
    for lr in learning_rates:
        dire = str(bs)+'_'+str(lr)+'/'
        os.mkdir(dire)
        model = cifar10vgg(bs=bs, lr=lr, epochs=100, save_path=dire)

