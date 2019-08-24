# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
def get_a_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(num_units=10) #也可以换成别的，比如GRUCell，BasicRNNCell等等
 
# X = tf.random_normal(shape=[4, 5, 6], dtype=tf.float32)
# X = tf.reshape(X, [-1, 5, 6])
X = np.array([[[1,2,3,4,6],
         [0,1,2,3,8],
         [3,6,8,1,2],
         [2,3,6,4,1]],
        [[2,3,5,6,8],
         [3,4,5,1,7],
         [6,5,9,0,2],
         [2,3,4,6,1]],
        [[2,3,5,1,6],
         [3,5,2,4,7],
         [4,5,2,4,1],
         [3,4,3,2,6]]])
X = tf.to_float(X)
 
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(1)], state_is_tuple=True)
# initial_state = stacked_lstm.zero_state(5, tf.float32)
# output, state = tf.nn.dynamic_rnn(stacked_lstm, X, initial_state=initial_state, time_major=True)
output, state = tf.nn.dynamic_rnn(stacked_lstm, X, time_major=False, dtype=tf.float32)
 
last = output[:, -1, :]  # 取最后一个时序输出作为结果
# fc_dense = tf.layers.dense(last, 10, name='fc1')
# fc_drop = tf.contrib.layers.dropout(fc_dense, 0.8)
# fc1 = tf.nn.relu(fc_drop)
 
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(last))
 
    print ('-------------------------\n')
 
    print (sess.run(state[0].h))