# Fashion MNIST数据集构建全连接神经网络模型
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print(train_images.shape)
# 构建神经元模型
model = keras.Sequential()
# 设置输入格式
model.add(keras.layers.Flatten(input_shape=(28, 28)))
# 添加神经元和激活函数
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# softmax把数字控制在0-1之间
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
train_images = train_images / 255
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3)
test_images = test_images / 255
model.evaluate(test_images, test_labels)
demo=tf.reshape(test_images[1],(1,28,28))
print(np.argmax(model.predict(demo)))
print(test_labels[1])
