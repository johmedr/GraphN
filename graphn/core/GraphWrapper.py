from ..utils._core_utils import _Wrapper

from .GraphShape import GraphShape

import keras.backend as K
import warnings
    

class GraphWrapper(_Wrapper):
    """
    A wrapper for a graph. Can be seen as a kind of list holding an adjacency 
    matrix as first element and a matrix of nodes features as second element. 
    Can be used as an input for GraphLayer objects. 
    Inherits from list, but the main list's methods are disabled (see _Wrapper 
    definition for more details). 

    Args: 
        - adjacency: a (..., N, N) tensor, 
        - nodes: a (..., N, F) tensor

    Properties: 
        - adjacency and nodes getters/setters,
        - n_nodes: the number of nodes N,
        - n_features: the length of nodes' features F

    A GraphWrapper object must be built, either by passing adjacency and nodes 
    as arguments when instanciating the object or by calling the 'build()' method 
    with adjacency and nodes. 
    """

    def __init__(self, nodes=None, adjacency=None, name="GraphWrapper"):
        super(GraphWrapper, self).__init__()

        self._keras_shape = None
        self._built = False

        self._nodes = None
        self._adjacency = None

        self._n_nodes = None
        self._n_features = None

        if nodes is not None and adjacency is not None: 
            self.build(nodes=nodes, adjacency=adjacency)

    def build(self, nodes=None, adjacency=None):
        """
        Checks the shapes and builds the GraphWrapper with the given 
        adjacency matrix and nodes. If nodes or adjacency matrix are 
        None, uses respectively self._nodes or self._adjacency to build 
        the object. Useful to automatically re-building the GraphWrapper
        when the nodes or adjacency setter is called. 

        Args: 
            - nodes: a (..., N, F) tensor,
            - adjacency: a (..., N, N) tensor

        Returns 'self'
        """
        if nodes is None and self._nodes is not None:
            nodes = self._nodes
        elif not K.is_tensor(nodes):
            raise ValueError("Nodes must be a tensor.")

        if adjacency is None and self._adjacency is not None:
            adjacency = self._adjacency
        else: 
            if not isinstance(adjacency, list): 
                _adjacency = [adjacency]
            else: 
                _adjacency = adjacency

            for a in _adjacency: 
                if  not K.is_tensor(a):
                    raise ValueError("Adjacency must be a tensor.")

        nodes_shape = K.int_shape(nodes)
        adjacency_shape = K.int_shape(adjacency)

        # Creating a GraphShape object will handle shape checking
        if self._keras_shape is None: 
            self._keras_shape = GraphShape(nodes_shape=nodes_shape, adjacency_shape=adjacency_shape)
        else: 
            self._keras_shape.build(nodes_shape=nodes_shape, adjacency_shape=adjacency_shape)

        self._nodes = nodes
        self._adjacency = adjacency

        self._n_features = nodes_shape[-1]
        self._n_nodes = nodes_shape[-2]

        super(GraphWrapper, self)._clear()
        super(GraphWrapper, self)._extend([self.nodes, self.adjacency])

        self._built = True

        return self

    def __str__(self):
        return "GraphWrapper< nodes:%s || adj:%s  >" % (self._nodes, self._adjacency)

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._built = False
        self.build(nodes=nodes)
        return self._nodes

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency):
        self._built = False
        self.build(adjacency=adjacency)
        return self._adjacency

    @property
    def shape(self):
        if not self._built:
            raise ValueError(
                "Tried to access the shape of a not-built GraphWrapper.")
        return self._keras_shape

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_nodes(self):
        return self._n_nodes


