import tensorflow as tf


a = tf.Variable(tf.random_normal(shape=[4,7,64]))
b = tf.Variable(tf.random_normal(shape=[4,64,1]))

res = tf.matmul(a,b)
