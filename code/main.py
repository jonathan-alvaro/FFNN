from layers import Dense
import numpy as np
import model

d = Dense(node_count=3, input_length=4)
print(d(np.matrix([[1, 2, 3, 4]])))
