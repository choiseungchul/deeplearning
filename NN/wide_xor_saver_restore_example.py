import tensorflow as tf
import numpy as np


sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./saved_vars/wide_xor_saved/train_data.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./saved_vars/wide_xor_saved'))

x_data = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)


# placeholder를 쓰기위해 X, Y를 지정
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

graph = tf.get_default_graph()

W1 = graph.get_tensor_by_name('weight1:0')
b1 = graph.get_tensor_by_name('bias1:0')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = graph.get_tensor_by_name('weight2:0')
b2 = graph.get_tensor_by_name('bias2:0')
hypo = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# Access saved Variables directly
# print(sess.run('weight1:0'))
# print(sess.run('weight2:0'))
# print(sess.run('bias1:0'))
# print(sess.run('bias2:0'))


pred = tf.cast(hypo > 0.5, dtype=tf.float32)
accu = tf.reduce_mean(tf.cast(tf.equal(pred, Y), dtype=tf.float32))

h, c, a = sess.run([hypo, pred, accu], feed_dict={X: x_data, Y:y_data})

print('h:',h, 'cost:',c, "accuracy:",a)

# This will print 2, which is the value of bias that we saved

#
# # Now, let's access and create placeholders variables and
# # create feed-dict to feed new data
#
# graph = tf.get_default_graph()
# w1 = graph.get_tensor_by_name("weight1:0")
# w2 = graph.get_tensor_by_name("weight2:0")
# feed_dict ={w1:13.0,w2:17.0}
#
# #Now, access the op that you want to run.
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
#
# print sess.run(op_to_restore,feed_dict)
#This will print 60 which is calculated