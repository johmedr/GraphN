import keras.backend as K 

from ..core import GraphWrapper, GraphLayer

class GraphConv(GraphLayer):
    """
    First-order approximation of a graph spectral convolution, as suggested in 
    https://arxiv.org/pdf/1609.02907.pdf
    """
    def __init__(self, n_filters, activation=K.tanh, name="GraphConv", kernel_regularizer=None, **kwargs):

        self.output_dim = n_filters
        self.activation = activation
        self.name = name
        self.kernel_regularizer = kernel_regularizer

        super(GraphConv, self).__init__(**kwargs)
        
    def build(self, input_shape): 
        """ Channel last: (batchs, n_nodes, n_features)"""
        assert(isinstance(input_shape, list))

        adj_shape, x_shape = input_shape
        
        assert len(adj_shape) >= 2, "Expected more than 2 dims, get %s"%(adj_shape,)
        assert len(x_shape) >= 2, "Expected more than 2 dims, get %s"%(x_shape,)

        self.kernel = self.add_weight(name='kernel', 
                                      shape=(x_shape[-1], self.output_dim),
                                      regularizer=self.kernel_regularizer, 
                                      initializer='uniform',
                                      trainable=True)

        super(GraphConv, self).build(input_shape)

    def call(self, x): 
        """ x: List containing 
              - the adjacency matrix (shape: (n_nodes, n_nodes))
              - the nodes features (shape: (n_nodes, n_features))"""
        if isinstance(x, list): 
            adjacency, nodes = x
        elif isinstance(x, GraphWrapper): 
            adjacency = x.adjacency
            nodes = x.nodes
        else: 
            raise ValueError()
        
        new_nodes = K.dot(nodes, self.kernel)
        if len(K.int_shape(nodes)) > 2: 
            new_nodes = K.batch_dot(adjacency, new_nodes)
        else: 
            new_nodes = K.dot(adjacency, new_nodes)

        return self.make_output_graph(
            adjacency=adjacency, nodes=self.activation(new_nodes))