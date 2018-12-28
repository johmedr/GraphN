import keras.backend as K

from ..core import GraphLayer


class GraphPoolingCell(GraphLayer):
    """ 
    Applies a kind of hierarchical pooling on a graph, 
    assigning adjacency matrix and nodes to a new graph configuration. 
    For more details, see the "Pooling with an assignement matrix" section 
    on page 4 in "Hierarchical Graph Representation Learning with
    Differentiable Pooling" (Rex Ying et al., 2018, https://arxiv.org/pdf/1806.08804.pdf). 
    """

    def __init__(self, output_dim, **kwargs):
        super(GraphPoolingCell, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        """ Channel last: (batchs, n_nodes, n_features)"""
        assert isinstance(input_shape, list) and len(input_shape) == 3

        x_shape, adj_shape, assignment_shape = input_shape

        assert len(adj_shape) >= 2, "Expected at least 2 dims, get %s" % (
            adj_shape,)
        assert len(x_shape) >= 2, "Expected at least 2 dims, get %s" % (
            x_shape,)
        assert len(assignment_shape) >= 2 and assignment_shape[-2] == adj_shape[-1] and assignment_shape[-1] == self.output_dim, \
            "Wrong assignment shape: get %s, expected the last 2 dims to be %s" % (
                assignment_shape, (adj_shape[-1], self.output_dim))

    def call(self, x):
        """ 
        x: List containing             
            - the adjacency matrix with shape (..., N, N)
            - the nodes with shape (..., N, F)
            - the assignment matrix with shape (N, N')
        NB: adjacency and nodes can be given in a GraphWrapper object

        Returns a graph wrapper holding 
            - the new adjacency matrix with shape (..., N', N')
            - the new nodes with shape (..., N', F)
        """
        if not (isinstance(x, list) and len(x) == 3):
            raise AttributeError(
                "Incorrect arguments for layer %s in call(). Get %s." % (self.name, x))

        nodes, adjacency, assignment = x

        adj_shape = K.int_shape(adjacency)
        x_shape = K.int_shape(nodes)

        assert len(adj_shape) <= 3 and len(
            x_shape) <= 3, "Not implemented for more than 3 dims (batch included)"

        assignment_transposed = K.transpose(assignment)

        #new_adj_shape = [-1 if i is None else i for i in adj_shape[:-2]] + [self.output_dim, self.output_dim]
        #new_nodes_shape = [-1 if i is None else i for i in x_shape[:-2]] + [self.output_dim, self._output_graph_wrapper.n_features]

        if len(adj_shape) > 2 and len(x_shape) > 2:
            new_adj_shape = [-1, self.output_dim, self.output_dim]
            new_nodes_shape = [-1, self.output_dim, x_shape[-1]]

            new_adj = K.dot(adjacency, assignment)
            new_adj = K.reshape(new_adj, [adj_shape[-1], -1])
            new_adj = K.dot(assignment_transposed, new_adj)
            new_adj = K.reshape(new_adj, new_adj_shape)

            new_nodes = K.reshape(nodes, [adj_shape[-1], -1])
            new_nodes = K.dot(assignment_transposed, new_nodes)
            new_nodes = K.reshape(new_nodes, new_nodes_shape)

        else:
            new_adj = K.dot(assignment_transposed,
                            K.dot(adjacency, assignment))
            new_nodes = K.dot(assignment_transposed, nodes)

        return self.make_output_graph(adjacency=new_adj, nodes=new_nodes)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(GraphPoolingCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
