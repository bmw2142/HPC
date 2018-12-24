# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:51:38 2018

@author: Ginger
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder


# placeholder
batch_size = 100
inputs = tf.placeholder(tf.float32, [batch_size,784])
targets = tf.placeholder(tf.int32, [batch_size])

# model
hidden1 = tf.layers.dense(inputs, 128, activation=tf.nn.relu, name='hidden1')
hidden2 = tf.layers.dense(hidden1, 32, activation=tf.nn.relu, name='hidden2')
logits = tf.layers.dense(hidden2, 10, activation=None, name='softmax_linear')

# loss + train_op
loss = tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=logits)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)




profiler = model_analyzer.Profiler(graph=sess.graph)


run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)


run_metadata = tf.RunMetadata()





mnist = input_data.read_data_sets(train_dir='./',fake_data=False)

feed_dict = dict()
for step in range(100):
    images_feed, labels_feed = mnist.train.next_batch(batch_size, fake_data=False)
    feed_dict = {inputs: images_feed, targets: labels_feed}
    #每 10 步，蒐集一下統計數據：
    if step % 10 == 0:
        _, loss_value = sess.run(fetches=[train_op, loss],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

        #將本步蒐集的統計數據添加到tfprofiler實例中     
        profiler.add_step(step=step, run_meta=run_metadata)
    else:
        _, loss_value = sess.run(fetches=[train_op, loss],
                               feed_dict=feed_dict)







profile_code_opt_builder = option_builder.ProfileOptionBuilder()

#過濾條件：顯示minist.py代碼。
profile_code_opt_builder.with_max_depth(1000)
profile_code_opt_builder.with_node_names(show_name_regexes=['mnist.py.*'])

#過濾條件：只顯示執行時間大於10us的代碼
profile_code_opt_builder.with_min_execution_time(min_micros=10)

#顯示字段：執行時間，且結果按照時間排序
profile_code_opt_builder.select(['micros'])
profile_code_opt_builder.order_by('micros')

#顯示視圖爲code view
profiler.profile_python(profile_code_opt_builder.build())







run_metadata = tf.RunMetadata()

_, loss_value = sess.run([train_op, loss],
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_metadata)

op_log = tfprof_log_pb2.OpLog()



tf.contrib.tfprof.tfprof_logger.write_op_log(
        tf.get_default_graph(),
        log_dir="/tmp/log_dir",
        op_log=op_log,
        run_meta=run_metadata)

tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_metadata=run_metadata,
        op_log=op_log,
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)









































