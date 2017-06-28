import tensorflow as tf
import numpy as np
import sys

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(len(mnist[0]))
print(len(mnist[1]))
print(len(mnist[2]))
