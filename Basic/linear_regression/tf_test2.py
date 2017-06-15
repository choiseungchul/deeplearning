import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# hypothesis
hypothesis = W * x_data + b

#cost function
# reduce_mean -> 평균
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#minimize
# cost가 거의 0 이 되는 지점을 찾는것이 목적이기 때문에 이 함수를 쓴다
a = tf.Variable(0.1) # Leaning rate 배움단계 ( 작을수록 더 정확하게 찾음 )
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print( step, sess.run(cost), sess.run(W), sess.run(b))


