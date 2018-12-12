import keras.backend as K 

from ..core import GraphWrapper, GraphLayer

class GraphPooling(GraphLayer):
    """ https://arxiv.org/pdf/1806.08804.pdf """
    def __init__(self, output_dim, trainable=True, **kwargs):
        self.output_dim = output_dim
        self.trainable = trainable
        self._output_graph_wrapper = None

        super(GraphPooling, self).__init__(**kwargs)
        
    def build(self, input_shape): 
        """ Channel last: (batchs, n_nodes, n_features)"""
        assert(isinstance(input_shape, list))

        adj_shape, x_shape = input_shape
        
        assert len(adj_shape) >= 2, "Expected more than 2 dims, get %s"%(adj_shape,)
        assert len(x_shape) >= 2, "Expected more than 2 dims, get %s"%(x_shape,)

        self._output_graph_wrapper = GraphWrapper(n_nodes=self.output_dim, n_features=x_shape[-1], name=self.name)
        
        self.assignment = self.add_weight(name='assignment', 
                                      shape=(adj_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=self.trainable)
        
        super(GraphPooling, self).build(input_shape)
        
    def call(self, x): 
        """ x: List containing 
              - the adjacency matrix (shape: (n_nodes, n_nodes))
              - the nodes features (shape: (n_nodes, n_features))"""
        if isinstance(x, list): 
            adjacency, nodes = x
        if isinstance(x, GraphWrapper): 
            adjacency = x.adjacency
            nodes = x.nodes

        n_nodes_input = K.int_shape(adjacency)[-1]

        assignment_transposed = K.transpose(self.assignment)
        print(assignment_transposed.shape)

        new_adj = K.dot(adjacency, self.assignment)
        new_adj = K.reshape(new_adj, [n_nodes_input, -1])
        new_adj = K.dot(assignment_transposed, new_adj)
        new_adj = K.reshape(new_adj, [-1, self._output_graph_wrapper.n_nodes, self._output_graph_wrapper.n_nodes])

        new_nodes = K.reshape(nodes, [n_nodes_input, -1])
        new_nodes = K.dot(assignment_transposed, new_nodes)
        new_nodes = K.reshape(new_nodes, [-1, self._output_graph_wrapper.n_nodes, self._output_graph_wrapper.n_features])

        self._output_graph_wrapper.build(
            [new_adj, new_nodes])

        return self._output_graph_wrapper
      
    def compute_output_shape(self, input_shape):
        return self._output_graph_wrapper.shape