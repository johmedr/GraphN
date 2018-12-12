from graphn.core import GraphWrapper
from graphn.layers import GraphConv
from graphn.layers import GraphPooling

import keras.backend as K 
from keras.layers import Input
from keras.models import Model

from keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def show_model(model): 
	plt.rcParams["figure.figsize"] = (25,25) # (w, h)
	plot_model(model, to_file='model.png', show_shapes=True)
	img = mpimg.imread('model.png')
	ax = plt.subplot(1,1,1)
	plt.imshow(img, aspect='equal')
	ax.grid(False)
	plt.show()

n_nodes = 15
n_features = 7
timesteps = 100

a = Input(shape=(n_nodes, n_nodes), name='adjacency')
x = Input(shape=(n_nodes, n_features), name='nodes')

g = GraphWrapper([a, x])
print(g.shape)

h = GraphConv(10)(g)
h = GraphPooling(5)(h)
h = GraphConv(20)(h)
y = GraphPooling(1)(h)

model = Model([a, x], y)
model.compile(optimizer='adam', loss='mse')

show_model(model)

print(model.summary(line_length=180))