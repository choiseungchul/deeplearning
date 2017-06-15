# 다중 변수에 대한 Linear regression

import tensorflow as tf

# x1_data = [1,0,3,0,5]
# x2_data = [0,2,0,4,0]
y_data = [1,2,3,4,5]

# x_data = [
#     [ 0.,2.,0.,4.,0.],
#     [1.,0.,3.,0.,5.]
# ]

x_data = [
    [1,1,1,1,1],
    [0.,2.,0.,4.,0.],
    [1.,0.,3.,0.,5.]
]

# W1 = tf.Variable( tf.random_uniform([1], -1.0, 1.0) )
# W2 = tf.Variable( tf.random_uniform([1], -1.0, 1.0) )
# W = tf.Variable(tf.random_uniform( [1,3], -1.0, 1.0 ))
# b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W = tf.Variable(tf.random_uniform( [1,3], -1.0, 1.0 ))

# hypothesis = W1 * x1_data + W2 * x2_data + b
# matrix multiplication
hypothesis = tf.matmul( W, x_data )

cost = tf.reduce_mean( tf.square( hypothesis - y_data ) )

learn_rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(learn_rate)
train = optimizer.minimize( cost )

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for step in range(3001):
    sess.run(train)
    if step % 20 == 0:
        print( step , sess.run(cost), sess.run(W) )










