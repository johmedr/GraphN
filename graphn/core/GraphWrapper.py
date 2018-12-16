import keras.backend as K

from ..utils._core_utils import _Wrapper


class GraphWrapper(_Wrapper):

    def __init__(self, adjacency=None, nodes=None, name="GraphWrapper"):
        """ 
                - adjacency with shape (?, n_nodes, n_nodes) 
                - nodes with shape (?, n_nodes, n_features)
        """
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

    def check_shape(self, adjacency_shape, nodes_shape):
        if len(adjacency_shape) < 2:
            raise ValueError("Expected at least 2 dims, get %s" %
                             (adjacency_shape,))

        if len(nodes_shape) < 2:
            raise ValueError("Expected at least 2 dims, get %s" %
                             (nodes_shape,))

        if adjacency_shape[-1] != adjacency_shape[-2]:
            raise ValueError("Wrong adjacency shape: get %s." %
                             (adjacency_shape[-2:]))

        # Let keras do the job
        # if nodes_shape[-2] != adjacency_shape[-1]:
        #    raise ValueError("Adjacency shape and nodes shape doesn't match. Get %s and %s."%(
        #        adjacency_shape[-2:], nodes_shape[-2:]))

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
