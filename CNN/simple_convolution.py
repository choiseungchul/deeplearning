import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#input, output 크기 계산
# N : 전체 크기
# F : 필터 크기
# stride : 찍을 간격
# output size = (N - F) / stride + 1
# e.g N = 7, F = 3
# stride 1 => (7-3)/1 + 1 = 5  5x5크기
# stride 2 => (7-3)/2 + 1 = 3  3x3크기
# padding 사용시 이미지 크기를 유지한다
# stride 1 => (9-3)/1 + 1 = 7
# stride 2 => (9-3)/2 + 1 = 4

# max pooling => sampling 이미지 => 값중 가장 높은 것을 지닌것으로 변환

# test image 생성
# datatype 을 정해줘야함
sess = tf.InteractiveSession()

image = np.array([[[[1.], [2.], [3.]],
                   [[4.], [5.], [6.]],
                   [[7.], [8.], [9.]]]], dtype=np.float32)


print(image.shape)

#plt.imshow(image.reshape(3,3), cmap='Greys')

# Simple convolution layer
# filter 2x2 stride = 1
# 필터 생성
weight = tf.constant([[[[1.]], [[1.]]],
                      [[[1.]], [[1.]]]], dtype=tf.float32)
print("weight shape", weight.shape)

# conv2d 변환, VALID = not padding, SAME = padding을 하기 때문에 같은 크기가 된다
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
#conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
conv2d_img = conv2d.eval()

print("conv2d_img.shape", conv2d_img.shape)


## No padding 출력값
#(1, 3, 3, 1)
#weight shape (2, 2, 1, 1) => 마지막 1 이 필터의 갯수
#conv2d_img.shape (1, 2, 2, 1)

## Padding 출력값
#(1, 3, 3, 1)
#weight shape (2, 2, 1, 1) => 마지막 1 이 필터의 갯수
#conv2d_img.shape (1, 3, 3, 1)




# 필터 여러개 정의







