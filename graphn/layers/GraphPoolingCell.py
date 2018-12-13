import keras.backend as K 

from ..core import GraphLayer

class GraphPoolingCell(GraphLayer):
    """ A simple pooling layer """
    def __init__(self, output_dim, kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.kernel_regularizer = kernel_regularizer 
        
        super(GraphPoolingCell, self).__init__(**kwargs)
        
    def build(self, input_shape): 
        """ Channel last: (batchs, n_nodes, n_features)"""
        assert isinstance(input_shape, list) and len(input_shape) == 3

        adj_shape, x_shape, assignment_shape = input_shape
        
        assert len(adj_shape) >= 2, "Expected more than 2 dims, get %s"%(adj_shape,)
        assert len(x_shape) >= 2, "Expected more than 2 dims, get %s"%(x_shape,)
        assert len(assignment_shape) >= 2 and assignment_shape[-2] == adj_shape[-1] and assignment_shape[-1] == self.output_dim, \
            "Wrong assignment shape: get %s, expected the last 2 dims to be %s"%(assignment_shape, (adj_shape[-1], self.output_dim))
        
        self.add_output_graph(n_nodes=self.output_dim, n_features=x_shape[-1], name=self.name)
        
        super(GraphPoolingCell, self).build(input_shape)
        
    def call(self, x): 
        """ x: List containing 
              - a graph wrapper (shape: [(n_nodes, n_nodes), (n_nodes, n_features)])
              - the assignment matrix (shape: (n_nodes, new_n_nodes))"""
        if not (isinstance(x, list) and len(x) == 3): 
            raise AttributeError("Incorrect arguments for layer %s in call(). Get %s."%(self.name, x)) 

        adjacency = x[0]
        nodes = x[1]
        assignment = x[2]

        adj_shape = K.int_shape(adjacency)
        x_shape = K.int_shape(nodes)

        assert len(adj_shape) <= 3 and len(x_shape) <= 3, "Not implemented for more than 3 dims (batch included)"

        assignment_transposed = K.transpose(assignment)

        #new_adj_shape = [-1 if i is None else i for i in adj_shape[:-2]] + [self.output_dim, self.output_dim]
        #new_nodes_shape = [-1 if i is None else i for i in x_shape[:-2]] + [self.output_dim, self._output_graph_wrapper.n_features]

        new_adj_shape = [-1, self.output_dim, self.output_dim]
        new_nodes_shape = [-1, self.output_dim, self._output_graph_wrapper.n_features]

        new_adj = K.dot(adjacency, assignment)
        new_adj = K.reshape(new_adj, [adj_shape[-1], -1])
        new_adj = K.dot(assignment_transposed, new_adj)
        new_adj = K.reshape(new_adj, new_adj_shape)

        new_nodes = K.reshape(nodes, [adj_shape[-1], -1])
        new_nodes = K.dot(assignment_transposed, new_nodes)
        new_nodes = K.reshape(new_nodes, new_nodes_shape)

        return self.make_output_graph(adjacency=new_adj, nodes=new_nodes)