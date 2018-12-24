import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

list_ = ["/cpu:0","/cpu:0","/cpu:0","/cpu:0","/cpu:0"]


tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]


tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

# CNN  list_[]
store = []

with tf.device(list_[0]):
   
    conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
        inputs=image,
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        name="conv1"
  

    )      

    # -> (28, 28, 16)
with tf.device(list_[1]):
    pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2

    )  
with tf.device(list_[2]):                 # -> (14, 14, 16)
    conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu,name="conv2")    # -> (14, 14, 32)
with tf.device(list_[3]):    
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
    flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
with tf.device(list_[4]):
    output = tf.layers.dense(flat, 10)              
  


loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)      
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(         
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
sess.run(init_op)     

##
profiler = model_analyzer.Profiler(graph=sess.graph)
run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
##


for step in range(300):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y},options=run_options, run_metadata=run_metadata)
        profiler.add_step(step=step, run_meta=run_metadata)
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)


##profiling excution

profile_code_opt_builder = option_builder.ProfileOptionBuilder()
profile_code_opt_builder.with_max_depth(1000)
profile_code_opt_builder.with_node_names(show_name_regexes=['mnist.py.*'])
profile_code_opt_builder.with_min_execution_time(min_micros=10)
profile_code_opt_builder.select(['micros'])
profile_code_opt_builder.order_by('micros')
profiler.profile_python(profile_code_opt_builder.build())



profile_op_opt_builder = option_builder.ProfileOptionBuilder()
profile_op_opt_builder.select(['micros','occurrence'])
profile_op_opt_builder.order_by('micros')
profile_op_opt_builder.with_max_depth(4)
profiler.profile_operations(profile_op_opt_builder.build())

profiler.advise(options=model_analyzer.ALL_ADVICE)