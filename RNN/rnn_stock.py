import tensorflow as tf
import numpy as np
import Util.Common as ut

timesteps = seq_len = 7
data_dim = 5
output_dim = 1

xy = np.loadtxt('test_data/data-stock-daily', delimiter=',')
xy = xy[::-1] # reverse order

xy = ut.MinMaxScaler(xy)

x = xy
y = xy[:,[-1]]

data_x = []
data_y = []

for i in range(0, len(y) - seq_len):
    print('')