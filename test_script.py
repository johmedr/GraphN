from graphn.core import GraphWrapper
from graphn.layers import GraphConv, GraphPoolingCell, GraphDiffPool

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

g = GraphWrapper(a, x)

print(g.shape)
 
pool1 = Input(shape=(15,5))
pool2 = Input(shape=(5,1))

h = GraphConv(10)(g)
print(h)
h = GraphDiffPool(5)(h)
print(h)
h = GraphConv(20)(h)
print(h)
y = GraphPoolingCell(1)([h, pool2])

model = Model([a, x, pool1, pool2], y)
model.compile(optimizer='adam', loss='mse')

show_model(model)

print(model.summary(line_length=180))