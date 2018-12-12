import keras.backend as K 
from keras.engine.topology import Layer

from ..core import GraphWrapper


class GraphPooling(Layer):
    """ https://arxiv.org/pdf/1806.08804.pdf """
    def __init__(self, output_dim, trainable=True, **kwargs):
        self.output_dim = output_dim
        self.trainable = trainable

        self._graph_wrapper = None

        super(GraphDiffPool, self).__init__(**kwargs)
        
    def build(self, input_shape): 
        """ Channel last: (batchs, n_nodes, n_features)"""
        assert(isinstance(input_shape, list))
        
        adj_shape, x_shape = input_shape
        assert(adj_shape[-1] == adj_shape[-2] == x_shape[-2])
        
        self.assignment = self.add_weight(name='assignment', 
                                      shape=(adj_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=self.trainable)
        
        super(GraphDiffPool, self).build(input_shape)
        
    def call(self, x): 
        """ x: List containing 
              - the adjacency matrix (shape: (n_nodes, n_nodes))
              - the nodes features (shape: (n_nodes, n_features))"""
        assert(isinstance(x, list))

        adjacency, nodes = x

        assignment_transposed = K.transpose(self.assignment)

        new_adj = K.dot(assignment_transposed, K.dot(adjacency, self.assignment))
        new_nodes = K.dot(assignment_transposed, nodes)


        return [new_adj, new_nodes]
      
    def compute_output_shape(self, input_shape):
        adj_shape, x_shape = input_shape
        return [adj_shape[:-2] + (self.output_dim, self.output_dim), 
                x_shape[:-2] + (self.output_dim, x_shape[-1])]
