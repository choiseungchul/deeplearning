import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

#[2] => [0,0,1,0,0,0,0] 으로 변경하기 위한 작업
nb_classes = 7
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# X 값과 matmul이 맞아야 하므로 16, one_hot 길이 만큼 
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
# bias 는 Y의 길이와 같으므로
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# 원래는 sigmoid 가 없을때는 hypothesis가 logits와 동일했다
# overfitting을 방지하기 위해 sigmoid 를 적용
# 다중 sigmoid 적용이 softmax
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# 원래는 복잡한 log 등의 식이 필요하지만 아래처럼 간단히 정리
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

# 경사로 내려가기 옵티마이저 실행
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 결과값 출력준비
# argmax는 hypothesis결과값중 가장 큰값 1개의 인덱스 번호를 리턴
# prediction은 학습이 완료된후 가설이 얼마나 맞는지 확인하기 위한것으로 가설의 값( 학습에 의해 계산된 값 )
prediction = tf.argmax(hypothesis,1 )
# 학습된가설에 의한값과 실제 학습데이터와 일치하는지 확인
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# 정확도를 %로 나타내기 위한값으로 성공/실패에 대한 평균을 낸다
accuracy = tf.reduce_mean(tf.cast( correct_prediction, tf.float32))


for step in range(2001):
    sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
    if step % 100 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data,Y:y_data})

        print( "step :", step, "\nCost: ", loss, "\nAccuracy: ", acc )


#학습 끝 , x데이터를 넣고 확인
# 학습한 데이터를 다시 넣어서 테스트 해본다고 보면됨
pred = sess.run(prediction, feed_dict={X:x_data})
# zip 이란
# [ [1],[2] ] 인 배열을 [ 1, 2 ] 로 만들어준다
for p, y in zip( pred, y_data.flatten() ):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


