from graphn.core import GraphWrapper
from graphn.layers import GraphConv
from graphn.layers import GraphPooling

import keras.backend as K 
from keras.layers import Input
from keras.models import Model

n_nodes = 15
n_features = 7

a = Input(shape=(n_nodes, n_nodes))
x = Input(shape=(n_nodes, n_features))

g = GraphWrapper([a, x])
print(g.shape)

h = GraphConv(10)(g)
h = GraphPooling(5)(h)
h = GraphConv(20)(h)
y = GraphPooling(1)(h)

model = Model([a, x], y)
model.compile(optimizer='adam', loss='mse')

print(model.summary(line_length=180))