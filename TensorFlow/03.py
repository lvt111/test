# 卷积神经网络 在全连接神经网络上增加四层
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 构建神经元模型
model = keras.Sequential()
# 卷积层
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
# 添加神经元和激活函数
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
# softmax把数字控制在0-1之间
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
train_images = train_images / 255
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=3)
test_images = test_images / 255
model.evaluate(train_images.reshape(-1, 28, 28, 1), train_images.reshape(-1, 28, 28, 1))
print(model.predict([[test_images[0]]]))
# print(test_labels[0])
