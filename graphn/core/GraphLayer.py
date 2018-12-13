from keras.engine.topology import Layer
from .GraphWrapper import GraphWrapper

class GraphLayer(Layer):
    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)
        self._output_graph_wrapper = None

    def add_output_graph(self, n_nodes, n_features, name=None): 
        if name is None: 
            name = "graph_wrapper_" + len(self._output_graph_wrapper.items())

        self._output_graph_wrapper = GraphWrapper(n_nodes=n_nodes, n_features=n_features, name=name)
        return self._output_graph_wrapper

    def make_output_graph(self, adjacency, nodes):
        if not self._built: 
            raise ValueError("The layer is not built yet. \
                Method add_output_graph() should be called in the build().")

        elif self._output_graph_wrapper is None:
            raise ValueError("There is no graph attached with the layer %s. \
                Method add_output_graph() should be called in the build()."%self.name)

        self._output_graph_wrapper.build(adjacency, nodes)
        return self._output_graph_wrapper

    def compute_output_shape(self, input_shape):
        if self._output_graph_wrapper is None: 
            raise ValueError("There is no graph attached with the layer %s. \
                Method add_output_graph() should be called in the build()."%self.name)

        return self._output_graph_wrapper.shape

    def __call__(self, inputs, **kwargs):
        outputs = super(GraphLayer, self).__call__(inputs, **kwargs)

        # Catch output and turn in into a graph wrapper  
        if isinstance(outputs, list) and len(outputs) == 2:
            print(outputs)
            outputs = self._output_graph_wrapper
            print(outputs)

        return outputs
        