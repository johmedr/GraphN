from ..core import GraphLayer
from ..core import GraphWrapper

from keras.layers import Dropout

from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton


class GraphDropout(GraphLayer):
    """ 
    Wraps a dropout on both adjacency matrix and nodes in a
    single layer. 
    Arguments with 'nodes_' and 'adjacency_' prefixes configure respectively 
    the nodes and adjacency matrix dropout layers. 
    """

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
        if isinstance(inputs, GraphWrapper):
            nodes = inputs.nodes
            adjacency = inputs.adjacency
        else:
            raise ValueError()

        new_nodes = Dropout(
            rate=self.nodes_rate,
            noise_shape=self.nodes_noise_shape,
            seed=self.nodes_seed
        )(nodes)

        new_adj = to_list(adjacency)
        if self.use_adjacency_dropout:
            new_adj = [Dropout(
                rate=self.adjacency_rate,
                noise_shape=self.adjacency_noise_shape,
                seed=self.nodes_seed
            )(a) for a in new_adj] 
            new_adj = unpack_singleton(new_adj)
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
