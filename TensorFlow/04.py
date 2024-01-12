# 自实现一个线性回归
import tensorflow as tf
# 准备数据
x = tf.random.normal(shape=[100, 1])
y_true = tf.matmul(x,[[0.8]]) + 0.7
# 构造模型
# 定义模型参数 用变量
weights = tf.Variable(initial_value=tf.random.normal(shape=[1, 1]))
bias = tf.Variable(initial_value=tf.random.normal(shape=[1, 1]))
y_predict = tf.reduce_mean(tf.square(x, weights)) + bias
# 构造损失函数
error = tf.reduce_mean(tf.square(y_predict - y_true))
# 优化损失
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
# 显示初始化变量
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print('训练前模型参数：权重%f，偏置%f' % (weights.eval(), bias.eval()))
    sess.run(optimizer)
    print('训练前模型参数：权重%f，偏置%f' % (weights.eval(), bias.eval()))
