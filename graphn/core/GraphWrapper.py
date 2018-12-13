import keras.backend as K

from ..utils._core_utils import _Wrapper


class GraphWrapper(_Wrapper): 
    def __init__(self, inputs=None, n_nodes=None, n_features=None, name="GraphWrapper"): 
        """ 
            inputs: a list [adjacency, nodes] with 
                - adjacency with shape (?, n_nodes, n_nodes) 
                - nodes with shape (?, n_nodes, n_features)
        """
        super(GraphWrapper, self).__init__() 

        self._keras_shape = None
        self._built = False

        self._nodes = None
        self._adjacency = None

        if n_nodes and n_features:
            self.n_nodes = int(n_nodes)
            self.n_features = int(n_features)

        elif inputs: 
            assert(isinstance(inputs, list) and len(inputs) == 2)

            A, x = inputs

            self.n_nodes = K.int_shape(A)[-1]
            self.n_features = K.int_shape(x)[-1]

            self.build(A, x)

        else: 
            raise AttributeError("You must provide either inputs or shapes to build the GraphWrapper.")

    def build(self, A, x): 
        shape = [K.int_shape(A), K.int_shape(x)]

        self.check_shape(shape)
        self._keras_shape = shape

        self._adjacency = A
        self._nodes = x

        super(GraphWrapper, self)._clear()
        super(GraphWrapper, self)._extend([self.adjacency, self.nodes])

        self._built = True


    def _check_adj_shape(self, adj_shape): 
        if len(adj_shape) < 2: 
            raise ValueError("Expected at least 2 dims, get %s"%(adj_shape,))

        if not (adj_shape[-1] ==  adj_shape[-2] == self.n_nodes): 
            raise ValueError("Wrong adjacency shape: got %s, expected %s."%(adj_shape[-2:], 
                (self.n_nodes, self.n_nodes)) )
          
        return True

    def _check_nodes_shape(self, nodes_shape): 
        if len(nodes_shape) < 2: 
            raise ValueError("Expected at least 2 dims, get %s"%(nodes_shape,))

        if not (nodes_shape[-1] == self.n_features and nodes_shape[-2] == self.n_nodes): 
            raise ValueError("Wrong nodes shape: got %s, expected %s."%(nodes_shape[-2:], 
                (self.n_nodes, self.n_features)) )
          
        return True 

    def check_shape(self, input_shape): 
        assert isinstance(input_shape, list) and len(input_shape) == 2

        adj_shape, nodes_shape = input_shape

        self._check_adj_shape(adj_shape)
        self._check_nodes_shape(nodes_shape)

        return True

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, x): 
        self._check_nodes_shape(K.int_shape(x))
        self._nodes = x

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, A): 
        self._check_adj_shape(K.int_shape(A))
        self._adjacency = A
    
    @property
    def shape(self):
        assert(self._built)
        return self._keras_shape

