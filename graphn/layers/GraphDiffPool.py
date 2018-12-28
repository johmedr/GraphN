import keras.backend as K

from keras import activations
from keras import regularizers
from keras import initializers
from keras import constraints

from .GraphPoolingCell import GraphPoolingCell
from .GraphConv import GraphConv

from ..core import GraphLayer
from ..utils import frobenius_norm
from ..utils import entropy


class GraphDiffPool(GraphLayer):
    """ 
    Applies a hierarchical pooling on a graph, predicting a cluster assignment 
    matrix transfering adjacency matrix and nodes to a new graph configuration. 
    For more details, see the "Hierarchical Graph Representation Learning with
    Differentiable Pooling" (Rex Ying et al., 2018, https://arxiv.org/pdf/1806.08804.pdf). 
    Args: 
        - output_dim: the number of nodes of the new graph,
        - gnn_embd_module: a graph neural network computing an embedding for the nodes,
        - gnn_pool_module: a graph neural network computing the cluster assignment matrix, 
        - other 'classical' args for a trainable keras layer 
    If gnn_embd_module or gnn_pool_module is None, the layer will build a GraphConv layer
    with the 'classical' configuration arguments. 
    """

    def __init__(self, output_dim,
                 gnn_embd_module=None,
                 gnn_pool_module=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphDiffPool, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.gnn_embd_module = gnn_embd_module
        self.gnn_pool_module = gnn_pool_module
        self.pooling_cell = None

    # Make it batch wise
    @staticmethod
    def pooling_loss(assignment, adjacency):
        assignment_transposed = K.transpose(assignment)
        return frobenius_norm(K.dot(assignment, assignment_transposed) - adjacency)

    @staticmethod
    def assignment_loss(assignment):
        return K.mean(entropy(assignment, axis=0))

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 2

        adj_shape, x_shape = input_shape

        assert len(adj_shape) >= 2, "Expected more than 2 dims, get %s" % (
            adj_shape,)
        assert len(x_shape) >= 2, "Expected more than 2 dims, get %s" % (
            x_shape,)

        # Build an embedding module if none is provided
        if not self.gnn_embd_module:
            self.gnn_embd_module = GraphConv(
                x_shape[-1],
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint
            )

        # Build a pooling module if none is provided
        if not self.gnn_pool_module:
            # Row-wise softmax
            self.gnn_pool_module = GraphConv(
                self.output_dim,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint
            )

        # Build a pooling cell to apply the assignment matrix to adjacency and
        # nodes
        self.pooling_cell = GraphPoolingCell(self.output_dim)

        self._built = True

    def call(self, x):
        if isinstance(x, list):
            adjacency, nodes = x
        else:
            raise ValueError()

        adj_shape = K.int_shape(adjacency)
        x_shape = K.int_shape(nodes)

        assert len(adj_shape) <= 3 and len(x_shape) <= 3, \
            "Not implemented for more than 3 dims (batch included)"

        # Compute embeddings for nodes
        embeddings = self.gnn_embd_module(x).nodes
        # Compute the adjacency matrix
        assignment = self.gnn_pool_module(x).nodes

        if len(adj_shape) > 2:
            assignment = K.reshape(
                assignment, [adj_shape[-1], self.output_dim])

        # Apply pooling to adjacency and embeddings
        pooled = self.pooling_cell([adjacency, embeddings, assignment])

        # Add the normalization losses
        self.add_loss([GraphDiffPool.pooling_loss(assignment, adjacency),
                       GraphDiffPool.assignment_loss(assignment)])

        return self.make_output_graph(adjacency=pooled.adjacency, nodes=pooled.nodes)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
