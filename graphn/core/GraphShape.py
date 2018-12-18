from ..utils import _Wrapper
import warnings

from keras.utils.generic_utils import to_list


class GraphShape(_Wrapper):

    def __init__(self, nodes_shape=None, adjacency_shape=None):
        self._adjacency_shape = None
        self._nodes_shape = None

        self.build(nodes_shape, adjacency_shape)

    def assert_graph_shape(self, nodes_shape, adjacency_shape):
        """
        Performs a simple check on adjacency and nodes, to 
        ensure that they are correct. 

        Please note that currently this will only result in a warning
        if adjacency and nodes don't hold the same number of nodes.  
        """
        if not isinstance(adjacency_shape, list):
            adjacency_shape = [adjacency_shape]

        if len(nodes_shape) < 2:
            raise ValueError("Expected at least 2 dims, get %s" %
                             (nodes_shape,))

        for shape in adjacency_shape:
            if len(shape) < 2:
                raise ValueError("Expected at least 2 dims, get %s" %
                                 (shape,))

            if shape[-1] != shape[-2]:
                raise ValueError("Wrong adjacency shape: get %s." %
                                 (shape[-2:],))

            # Let keras do the job, just raise a warning
            if nodes_shape[-2] != shape[-1]:
                warnings.warn("Adjacency shape and nodes shape doesn't match. Get %s and %s." % (
                    shape[-2:], nodes_shape[-2:]))

    def build(self, nodes_shape, adjacency_shape):
        self.assert_graph_shape(nodes_shape, adjacency_shape)

        self._nodes_shape = nodes_shape
        self._adjacency_shape = adjacency_shape

        self._clear()
        self._extend([self._nodes_shape] + to_list(self._adjacency_shape))

    @property
    def adjacency_shape(self):
        return self._adjacency_shape

    @adjacency_shape.setter
    def adjacency_shape(self, adjacency_shape):
        self.build(self.nodes_shape, adjacency_shape)
        return self._adjacency_shape

    @property
    def nodes_shape(self):
        return self._nodes_shape

    @nodes_shape.setter
    def nodes_shape(self, nodes_shape):
        self.build(nodes_shape, self.adjacency_shape)
        return self._nodes_shape

    def __str__(self):
        return 'GraphShape<%s>' % (super(GraphShape, self).__str__())
