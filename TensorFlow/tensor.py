import tensorflow as tf
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
# 创建张量
#x = tf.constant(4,shape=(1,1),dytpe=tf.float32)
x = tf.constant([[1,2,3],[4,5,6]])
print(x)
y=tf.zeros((3,3))
print(y)
z=tf.eye(3)
print(z)