from ..core import GraphLayer

from keras.layers import Dropout

class GraphDropout(GraphLayer):

    def __init__(self, 
                 nodes_rate, 
                 nodes_noise_shape=None, 
                 nodes_seed=None, 
                 adjacency_rate=None, 
                 adjacency_noise_shape=None, 
                 adjacency_seed=None, 
                 **kwargs):
        super(GraphDropout, self).__init__(**kwargs)

        self.nodes_rate = min(1., max(0., nodes_rate))
        self.nodes_noise_shape = nodes_noise_shape
        self.nodes_seed = nodes_seed
        self.supports_masking = True

        if adjacency_rate is not None: 
            self.use_adjacency_dropout = True
            self.adjacency_rate = min(1., max(0., adjacency_rate))
            self.adjacency_noise_shape = adjacency_noise_shape
            self.adjacency_seed = adjacency_seed
            self.supports_masking = True
        else: 
            self.use_adjacency_dropout = False

    def call(self, inputs):
        if isinstance(inputs, list):
            adjacency, nodes = inputs
        else:
            raise ValueError()

        new_nodes = Dropout(
            rate=self.nodes_rate, 
            noise_shape=self.nodes_noise_shape, 
            seed=self.nodes_seed
        )(nodes)

        if self.use_adjacency_dropout: 
            new_adj = Dropout(
                rate=self.adjacency_rate, 
                noise_shape=self.adjacency_noise_shape, 
                seed=self.nodes_seed
            )(adjacency)
        else: 
            new_adj = adjacency

        return self.make_output_graph(adjacency=new_adj, nodes=new_nodes)

    def get_config(self):
        config = {
            'nodes_rate': self.nodes_rate,
            'nodes_noise_shape': self.nodes_noise_shape,
            'nodes_seed': self.nodes_seed,
            'adjacency_rate': self.adjacency_rate,
            'adjacency_noise_shape': self.adjacency_noise_shape,
            'adjacency_seed': self.adjacency_seed
        }
        base_config = super(GraphDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))