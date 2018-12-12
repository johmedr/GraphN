import keras.backend as K

class GraphWrapper(list): 
    def __init__(self, inputs=None, n_nodes=None, n_features=None, name="GraphWrapper"): 
        """ 
            inputs: a list [adjacency, nodes] with 
                - adjacency with shape (?, n_nodes, n_nodes) 
                - nodes with shape (?, n_nodes, n_features)
        """
        super(GraphWrapper, self).__init__() 

        self._keras_shape = None
        self._built = False

        self.nodes = None
        self.adjacency = None

        if n_nodes and n_features:
            self.n_nodes = int(n_nodes)
            self.n_features = int(n_features)

        elif inputs: 
            assert(isinstance(inputs, list) and len(inputs) == 2)

            A, x = inputs

            self.n_nodes = K.int_shape(A)[-1]
            self.n_features = K.int_shape(x)[-1]

            self.build(inputs)

        else: 
            raise AttributeError("You must provide either inputs or shapes to build the GraphWrapper.")

    def build(self, inputs): 
        assert(isinstance(inputs, list) and len(inputs) == 2)
        
        A, x = inputs

        shape = [K.int_shape(A), K.int_shape(x)]
        self.check_shape(shape)
        self._keras_shape = shape

        self.adjacency = A
        self.nodes = x

        super(GraphWrapper, self).clear()
        super(GraphWrapper, self).extend([self.adjacency, self.nodes])

        self._built = True

    def check_shape(self, input_shape): 
        assert(isinstance(input_shape, list))

        adj_shape, x_shape = input_shape
        
        assert len(adj_shape) >= 2, "Expected more than 2 dims, get %s"%(adj_shape,)
        assert len(x_shape) >= 2, "Expected more than 2 dims, get %s"%(x_shape,)

        assert adj_shape[-1] ==  adj_shape[-2] == x_shape[-2] == self.n_nodes, \
            "Wrong shapes A: %s, x: %s for %d nodes."%(adj_shape[-2:], x_shape[-2], self.n_nodes)

        assert x_shape[-1] == self.n_features, \
            "Wrong shape x: %s for %d features."%(x_shape[-1], self.n_features) 

    def get_shape(self):
        assert(self._built)

        return self._keras_shape

    shape = property(get_shape)