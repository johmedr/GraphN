from __future__ import print_function

from graphn.datasets import cora
from graphn.layers import GraphConv, GraphDropout
from graphn.core import GraphWrapper

import time

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input

SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience
FILTER = 'chebyshev'
MAX_DEGREE = 2

X, A, y = cora.load_data()
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = cora.get_splits(
    y)

# Normalize nodes' inputs 
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = cora.preprocess_adj(A, SYM_NORM)
    A_ = A_.todense()
    support = 1

    # Graph holds data from cora 
    graph = [X, A_]
    G = Input(shape=(None, None), batch_shape=(None, None))

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = cora.normalized_laplacian(A, SYM_NORM)
    L_scaled = cora.rescale_laplacian(L)
    T_k = cora.chebyshev_polynomial(L_scaled, MAX_DEGREE)
    T_k = [tk.todense() for tk in T_k]
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    G = [Input(shape=(None, None), batch_shape=(None, None)) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

# Build 2 Inputs, one for the adjacency matrix, one for the nodes 
# G = Input(shape=(None, None), batch_shape=(None, None), name="adjacency")
X_in = Input(shape=(X.shape[1],), name="nodes")

# Wrap in a GraphWrapper
g = GraphWrapper(adjacency=G, nodes=X_in)
print(g)

# Apply dropout on nodes
H = GraphDropout(nodes_rate=0.5)(g)

# Apply a GraphConv with 16 filters 
H = GraphConv(16, activation='relu')(H)
H = GraphDropout(nodes_rate=0.5)(H)

# Apply a GraphConv with softmax to predict each nodes' class 
Y = GraphConv(y.shape[1], activation='softmax')(H)

# Build a model taking a graph in input and outputing nodes' classes 
model = Model(inputs=g, outputs=Y.nodes)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))

# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999

# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration (we mask nodes without labels for loss
    # calculation)
    model.fit(graph, y_train, sample_weight=train_mask,
              batch_size=A.shape[0], epochs=1, shuffle=False, verbose=0)

    # Predict on full dataset
    preds = model.predict(graph, batch_size=A.shape[0])

    # Train / validation scores
    train_val_loss, train_val_acc = cora.evaluate_preds(preds, [y_train, y_val],
                                                   [idx_train, idx_val])
    print("Epoch: {:04d}".format(epoch),
          "train_loss= {:.4f}".format(train_val_loss[0]),
          "train_acc= {:.4f}".format(train_val_acc[0]),
          "val_loss= {:.4f}".format(train_val_loss[1]),
          "val_acc= {:.4f}".format(train_val_acc[1]),
          "time= {:.4f}".format(time.time() - t))

    # Early stopping
    if train_val_loss[1] < best_val_loss:
        best_val_loss = train_val_loss[1]
        wait = 0
    else:
        if wait >= PATIENCE:
            print('Epoch {}: early stopping'.format(epoch))
            break
        wait += 1

# Testing
test_loss, test_acc = cora.evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))
