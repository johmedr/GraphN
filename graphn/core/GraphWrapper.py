from ..utils._core_utils import _Wrapper

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

    def __init__(self, adjacency=None, nodes=None, name="GraphWrapper"):
        super(GraphWrapper, self).__init__()

        self._keras_shape = None
        self._built = False

        self._nodes = None
        self._adjacency = None

        self._n_nodes = None
        self._n_features = None

        if K.is_tensor(nodes) and K.is_tensor(adjacency):
            self.build(adjacency, nodes)

        elif nodes is not None or adjacency is not None:
            raise ValueError("Adjacency and nodes must be tensors.")

    def build(self, adjacency=None, nodes=None):
        """
        Checks the shapes and builds the GraphWrapper with the given 
        adjacency matrix and nodes. If nodes or adjacency matrix are 
        None, uses respectively self._nodes or self._adjacency to build 
        the object. Useful to automatically re-building the GraphWrapper
        when the nodes or adjacency setter is called. 

        Args: 
            - adjacency: a (..., N, N) tensor, 
            - nodes: a (..., N, F) tensor

        Returns 'self'
        """
        if adjacency is None:
            adjacency = self._adjacency
        elif not K.is_tensor(adjacency):
            raise ValueError("Adjacency must be a tensor.")

        if nodes is None:
            nodes = self._nodes
        elif not K.is_tensor(nodes):
            raise ValueError("Nodes must be a tensor.")

        adjacency_shape = K.int_shape(adjacency)
        nodes_shape = K.int_shape(nodes)

        self.check_shape(adjacency_shape, nodes_shape)

        self._adjacency = adjacency
        self._nodes = nodes

        self._n_nodes = adjacency_shape[-1]
        self._n_features = nodes_shape[-1]

        self._keras_shape = [adjacency_shape, nodes_shape]

        super(GraphWrapper, self)._clear()
        super(GraphWrapper, self)._extend([self.adjacency, self.nodes])

        self._built = True

        return self

    def check_shape(self, adjacency_shape, nodes_shape):
        """
        Performs a simple check on adjacency and nodes, to 
        ensure that they are correct. 

        Please note that currently this will only result in a warning
        if adjacency and nodes don't hold the same number of nodes.  
        """
        if len(adjacency_shape) < 2:
            raise ValueError("Expected at least 2 dims, get %s" %
                             (adjacency_shape,))

        if len(nodes_shape) < 2:
            raise ValueError("Expected at least 2 dims, get %s" %
                             (nodes_shape,))

        if adjacency_shape[-1] != adjacency_shape[-2]:
            raise ValueError("Wrong adjacency shape: get %s." %
                             (adjacency_shape[-2:]))

        # Let keras do the job, just raise a warning
        if nodes_shape[-2] != adjacency_shape[-1]:
           warnings.warn("Adjacency shape and nodes shape doesn't match. Get %s and %s."%(
               adjacency_shape[-2:], nodes_shape[-2:]))

        return True

    def __str__(self):
        return "GraphWrapper< adj:%s || nodes:%s >" % (self._adjacency, self._nodes)

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
