from graphn.datasets import dd

from keras.layers import Input
import keras.backend as K
from keras.models import Model

from graphn.core import GraphWrapper
from graphn.layers import GraphDiffPool
from graphn.layers import GraphPoolingCell

nodes, train_data, test_data, n, N, m = dd.load_data()

train_adjs, train_lbls = train_data
test_adjs, test_lbls = test_data

# for i in range(len(train_adjs)):
#     dd.normalize_adj(train_adjs[i])

# for i in range(len(test_adjs)):
#     dd.normalize_adj(test_adjs[i])

# Build 2 Inputs, one for the adjacency matrix, one for the nodes
A_in = Input(shape=(n, n), batch_shape=(n, n), name="adjacency")
X_in = Input(shape=(n, nodes.shape[1]), batch_shape=(
    n, nodes.shape[1]), name="nodes")

# Wrap in a GraphWrapper
g = GraphWrapper(adjacency=A_in, nodes=X_in)

# Pool to 25% of the nodes
H = GraphDiffPool(output_dim=int(0.25 * n))(g)
print(type(H))
# Pool to 10% of the nodes
H = GraphDiffPool(output_dim=int(0.10 * n))(H)

# Final graph embedding (one final cluster)
y = GraphPoolingCell(output_dim=1)(H + [K.ones((H.n_nodes, 1))])

model = Model(g, y)
model.compile(optimizer='adam')

print(model.summary())
