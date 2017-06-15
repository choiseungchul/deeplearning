import numpy as np
import pprint
import tensorflow as tf

t = np.array([0.,1.,2.,3.,4.,5.,6.])

pp = pprint.PrettyPrinter(indent=4)

pp.pprint(t)

print(t.ndim)           # 깊이
print(t.shape)          # 배열의 전체적인 모양
print(t[0:4], t[1:3], t[2:-1])
print(t[:2], t[3:])


# 2D array
t = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.], [7.,8.,9.]])

pp.pprint(t)
print(t.shape)


# shape rank axis 의 정의

sess = tf.Session()

t = tf.constant([[1,2],[3,4]])
# shape 2,2 rank = 2 ( 맨앞 대괄호의 수로 바로 알수있다 )
# 구할때는 랭크 => 쉐잎 순으로 한다
# 쉐잎을 구할때는 맨 안쪽꺼 부터 구하는것이 좋다 
# axis = 0 부터 시작된다 -> 2차원배열일 경우 0, 1 의 축이 있다
# axis = -1 은 가장 안쪽의 축을 뜻한다
temp = tf.shape(t).eval(session=sess)
print(temp)



# 부록같은 내용이지만.. broadcasting 샘플
matrix1 = tf.constant([1., 2.])
matrix2 = tf.constant(3.) # 3. 이지만 연산시 자동으로 [3.,3.] 으로 변경된다

temp = (matrix1 + matrix2).eval(session=sess)
print(temp)


matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([3., 4.]) # 자동으로 랭크가 같아지고 더해진다

temp = (matrix1 + matrix2).eval(session=sess)
print(temp)

# tf.squeeze, tf.expand_dims 가끔 나오는것들이니 참고

# squeeze 는 배열의 형태를 펴준다?
temp = tf.squeeze([[0], [1], [2]]).eval(session=sess)
print(temp)

# expand_dims 는 rank를 하나더 늘려준다
temp = tf.expand_dims([0, 1, 2], 1).eval(session=sess)
print(temp)


# one_hot => 자주 쓰는함수이니 꼭 알아둘것
# classification 에서 씀
# class의 갯수에 따라 depth에 변화를 준다
temp = tf.one_hot([[0],[1],[2]], depth=3).eval(session=sess)
print(temp)

# cast
# prediction(예측값) 과 y(실제값) 비교시 많이 씀
temp = tf.cast([1.8, 2.2, 3.3, 4.5], tf.int32).eval(session=sess)
print(temp)
temp = tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval(session=sess)
print(temp)

# stack
# 여러개의 데이터를 쌓아준다
_x = [1, 4]
_y = [3, 6]
_z = [7, 8]

temp = tf.stack([_x, _y, _z]).eval(session=sess)
pp.pprint(temp)

# axis 옵션을 주게되면 axis의 기준으로 쌓는다
temp = tf.stack([_x, _y, _z], axis=1).eval(session=sess)
pp.pprint(temp)


# ones_like , zeros_like
# 배열의 모든 element들을 1, 0 으로 바꾼다
dt = [[1,2,3],[3,4,5]]
temp = tf.ones_like(dt).eval(session=sess)
print(temp)
temp = tf.zeros_like(dt).eval(session=sess)
print(temp)

# zip
for x, y in zip([1,2,3], [4,5,6]):
    print(x,y)
'''
1 4
2 5
3 6
'''
for x, y, z in zip([1, 2, 3], [4, 5, 6], [0, 1, 9]):
    print(x, y, z)
'''
1 4 0
2 5 1
3 6 9
'''