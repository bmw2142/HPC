import tensorflow as tf
import numpy as np

from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

##inport data
xt = np.load("xt.npy")
yi = np.load("yi.npy")
y_train = np.load("y_train.npy").reshape((-1,1))
##input output setting

tf_xt = tf.placeholder(tf.float32, [None, xt.shape[1] , xt.shape[2]])
tf_yi = tf.placeholder(tf.float32, [None, yi.shape[1] , yi.shape[2]])
tf_y = tf.placeholder(tf.int32, [None, 1]) 
tf_is_training = tf.placeholder(tf.bool, None)

##CSI model (CSI=Capture+Score+Integrate)

##%Capture
capture_x_bar = tf.layers.dense(tf_xt, 64, tf.nn.tanh)
dropout_Wa = tf.layers.dropout(capture_x_bar, rate=0.2, training=tf_is_training)
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
capture_lstm_ht, (h_c, h_n) = tf.nn.dynamic_rnn(rnn_cell,dropout_Wa,initial_state=None,dtype=tf.float32,time_major=False)
vj_0 = tf.layers.dense(h_n,16, tf.nn.tanh)
vj = tf.layers.dropout(vj_0, rate=0.2, training=tf_is_training)

##%CI model
#Li = tf.layers.dense(vj, 1, tf.nn.sigmoid)

##%Score
yi_bar = tf.layers.dense(tf_yi, 5, tf.nn.tanh)
si = tf.layers.dense(tf_yi, 1, tf.nn.sigmoid)
si_Flatten = tf.layers.Flatten()(si)
pj = tf.layers.dense(si_Flatten, 1)

##%Integrate
ci = tf.keras.layers.concatenate([vj,pj],axis=-1)
Li = tf.layers.dense(ci, 1, tf.nn.sigmoid)

##Loss Function Setting
loss_function = tf.losses.sigmoid_cross_entropy(tf_y, logits=Li)

##Optimizor Setting
optimzor = tf.train.AdamOptimizer(0.001).minimize(loss_function)

##Other Setting
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(Li, axis=1),)[1]

##Session Setting
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

##
profiler = model_analyzer.Profiler(graph=sess.graph)
run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
##

##Interal
for step in range(200):    # training
    _, loss_ = sess.run([optimzor, loss_function], {tf_xt: xt ,tf_yi:yi,tf_y: y_train, tf_is_training: True})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_xt: xt,tf_yi:yi ,tf_y: y_train, tf_is_training: True},options=run_options, run_metadata=run_metadata)
        profiler.add_step(step=step, run_meta=run_metadata)
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

##profiling excution

profile_code_opt_builder = option_builder.ProfileOptionBuilder()
profile_code_opt_builder.with_max_depth(1000)
profile_code_opt_builder.with_node_names(show_name_regexes=['mnist.py.*'])
profile_code_opt_builder.with_min_execution_time(min_micros=10)
profile_code_opt_builder.select(['micros'])
profile_code_opt_builder.order_by('micros')
profiler.profile_python(profile_code_opt_builder.build())