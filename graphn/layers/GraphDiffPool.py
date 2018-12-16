import keras.backend as K

from .GraphPoolingCell import GraphPoolingCell
from .GraphConv import GraphConv
from ..core import GraphLayer, GraphWrapper
from ..utils import frobenius_norm, entropy


class GraphDiffPool(GraphLayer):
    # TODO : parse args, activation, gnn layer and repetition, modules... Here is a minimal version
    # Makes the regularization 
    def __init__(self, output_dim, gnn_embd_module=None, gnn_pool_module=None, **kwargs):
        self.output_dim = output_dim
        self._gnn_embd_module = gnn_embd_module
        self._gnn_pool_module = gnn_pool_module
        self._pooling_cell = None
        super(GraphDiffPool, self).__init__(**kwargs)

    # Make it batch wise
    @staticmethod
    def pooling_loss(assignment, adjacency): 
        assignment_transposed = K.transpose(assignment)
        return frobenius_norm(K.dot(assignment, assignment_transposed) - adjacency)

    @staticmethod
    def assignment_loss(assignment): 
        return entropy(assignment, axis=0)

    def build(self, input_shape): 
        assert isinstance(input_shape, list) and len(input_shape) == 2

        adj_shape, x_shape = input_shape
        
        assert len(adj_shape) >= 2, "Expected more than 2 dims, get %s"%(adj_shape,)
        assert len(x_shape) >= 2, "Expected more than 2 dims, get %s"%(x_shape,)

        if not self._gnn_embd_module: 
            self._gnn_embd_module = GraphConv(x_shape[-1])

        if not self._gnn_pool_module: 
            # Row-wise softmax
            self._gnn_pool_module = GraphConv(self.output_dim, activation=lambda x: K.softmax(x, axis=-2))

        self._pooling_cell = GraphPoolingCell(self.output_dim)

        super(GraphDiffPool, self).build(input_shape)

    def call(self, x):
        if isinstance(x, list): 
            adjacency, nodes = x
        elif isinstance(x, GraphWrapper): 
            adjacency = x.adjacency
            nodes = x.nodes
        else: 
            raise ValueError()

        adj_shape = K.int_shape(adjacency)
        x_shape = K.int_shape(nodes)

        assert len(adj_shape) <= 3 and len(x_shape) <= 3, "Not implemented for more than 3 dims (batch included)"

        embeddings = self._gnn_embd_module(x)
        assignment = self._gnn_pool_module(x).nodes
        
        if len(adj_shape) > 2: 
            assignment = K.reshape(assignment, [adj_shape[-1], self.output_dim])

        pooled = self._pooling_cell([embeddings, assignment])

        self.add_loss([GraphDiffPool.pooling_loss(assignment, adjacency), 
            GraphDiffPool.assignment_loss(assignment)])

        return self.make_output_graph(adjacency=pooled.adjacency, nodes=pooled.nodes)









