from tensorflow import keras
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# 优化方案
model.compile(optimizer='sgd', loss='mean_squared_error')
x = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10,
     10, 10, 11, 11, 11, 11, 12, 12, 12, 12]
y = [79, 58, 54, 67, 95, 129, 76, 98, 44, 68, 73, 51, 14, 71, 44, 275, 148, 107, 138, 67, 87, 119, 146, 63, 23, 59, 72,
     40, 42, 50, 30, 50, 49, 46, 41, 31, 33, 44, 26, 11, 38, 65, 40, 25, 43, 68, 63, 36]
xs = np.array(x, dtype=int)
ys = np.array(y, dtype=int)
# 学习5次
model.fit(xs, ys, epochs=4000)
y = model.predict([1])
print(y)
