import tensorflow as tf



def build_rnn(seq_len, batch_size, state_size, num_feats):

  x = tf.placeholder(shape=[batch_size, seq_len, num_feats], dtype=tf.float32)
  tf.add_to_collection("x", x)
  y = tf.placeholder(dtype=tf.float32, shape=[batch_size * seq_len, 1])

  cell = tf.nn.rnn_cell.GRUCell(state_size)
  init_state = tf.get_variable('init_state', [batch_size, state_size],
  			       initializer=tf.constant_initializer(0.0))
  rnn_outputs, final_state = tf.nn.dynamic_rnn(cell,
					       x,
					       initial_state=init_state)

  W = tf.get_variable("affline_w", [state_size, 1], 
		      initializer=tf.random_normal_initializer())
  b = tf.get_variable("affline_b", [batch_size * seq_len, 1], 
		      initializer=tf.random_normal_initializer())
  print(rnn_outputs.get_shape().as_list())
  rnn_output_flat = tf.reshape(rnn_outputs, [batch_size * seq_len, state_size])
  print(rnn_output_flat.get_shape().as_list())

  logits = tf.matmul(rnn_output_flat,W) + b
#  logits_sm = tf.nn.softmax(logits)
  logits_sm = logits
  tf.add_to_collection('pred', logits_sm)
  print(logits.get_shape(), logits_sm.get_shape())
  loss = tf.reduce_mean(tf.square(logits_sm - y))

  adam = tf.train.AdamOptimizer(1e-5)
  train_op = adam.minimize(loss)
  return x, y, loss, train_op

#  W = tf.get_variabl

def build_multi_rnn(seq_len, batch_size, state_size, num_feats):

  x = tf.placeholder(shape=[batch_size, seq_len, num_feats], dtype=tf.float32)
  tf.add_to_collection("x", x)
  y = tf.placeholder(dtype=tf.float32, shape=[batch_size * seq_len, 1])

  cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
  init_state = tf.get_variable('init_state', [batch_size, state_size],
  			       initializer=tf.constant_initializer(0.0))
  rnn_outputs, final_state = tf.nn.dynamic_rnn(cell,
					       x,
						dtype=tf.float32)

  W = tf.get_variable("affline_w", [state_size, 1], 
		      initializer=tf.random_normal_initializer())
  b = tf.get_variable("affline_b", [batch_size * seq_len, 1], 
		      initializer=tf.random_normal_initializer())
  print(rnn_outputs.get_shape().as_list())
  rnn_output_flat = tf.reshape(rnn_outputs, [batch_size * seq_len, state_size])
  print(rnn_output_flat.get_shape().as_list())

  logits = tf.matmul(rnn_output_flat,W) + b
#  logits_sm = tf.nn.softmax(logits)
  logits_sm = logits
  tf.add_to_collection('pred', logits_sm)
  print(logits.get_shape(), logits_sm.get_shape())
  loss = tf.reduce_mean(tf.square(logits_sm - y))

  adam = tf.train.AdamOptimizer(1e-5)
  train_op = adam.minimize(loss)
  return x, y, loss, train_op

def build_logreg(seq_len, batch_size, state_size, num_feats, num_classes=3):
  x = tf.placeholder(shape=[batch_size, seq_len, num_feats], dtype=tf.float32)
  tf.add_to_collection("x", x)
  y = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_classes])
  x_flat = tf.reshape(x, [batch_size, seq_len * num_feats])

  W = tf.get_variable("w", [seq_len* num_feats, num_classes], 
		      initializer=tf.random_normal_initializer())
  b = tf.get_variable("affline_b", [batch_size, num_classes], 
		      initializer=tf.random_normal_initializer())

  logits = tf.matmul(x_flat,W) + b
#  logits_sm = tf.nn.softmax(logits)
  logits_sm = logits
  tf.add_to_collection('pred', logits_sm)
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_sm)
#  loss = tf.reduce_mean(tf.square(logits_sm - y))

  adam = tf.train.AdamOptimizer(1e-5)
  train_op = adam.minimize(loss)
  return x, y, loss, train_op

