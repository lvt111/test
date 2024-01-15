import tensorflow as tf

# 自定义图
new_g = tf.Graph()
with new_g.as_default():
    a = tf.constant(20)
    b = tf.constant(30)
    c = a + b
    print("c:\n", c)
    print("a 的图属性: \n", a.graph)
    print("c的图属性: \n", c.graph)

# 开启会话
with tf.compat.v1.Session(graph=new_g) as sess:
    c_t_value = sess.run(c)
    print('c_t: \n', c_t_value)
    print('sess的图属性： \n', sess.graph)
    tf.summary.create_file_writer('./TensorFlow/summary', graph=sess.graph)
