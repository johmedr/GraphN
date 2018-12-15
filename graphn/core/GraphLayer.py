from keras.engine.topology import Layer
from itertools import chain

from .GraphWrapper import GraphWrapper

class GraphLayer(Layer):

    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)
        self._output_graph_wrapper = GraphWrapper(name="graph_wrapper_%s"%self.name)

    def make_output_graph(self, adjacency, nodes):
        if not self._built: 
            raise ValueError("The layer is not built yet. \
                Method add_output_graph() should be called in the build().")

        elif self._output_graph_wrapper is None:
            raise ValueError("There is no graph attached with the layer %s. \
                Method add_output_graph() should be called in the build()."%self.name)

        self._output_graph_wrapper.build(adjacency=adjacency, nodes=nodes)
        return self._output_graph_wrapper

    def compute_output_shape(self, input_shape):
        return self._output_graph_wrapper.shape

    def __call__(self, inputs, **kwargs):
        
        # Keras do not supports nested lists, unpack nodes and adjacency
        if isinstance(inputs, list) and not isinstance(inputs, GraphWrapper): 
            _inputs = list(chain.from_iterable([i for i in inputs if isinstance(i, GraphWrapper)]))
            _inputs += [i for i in inputs if not isinstance(i, GraphWrapper)]

        else: 
            _inputs = inputs

        outputs = super(GraphLayer, self).__call__(_inputs, **kwargs)

        # Catch outputs and make sure to return a graph wrapper
        if isinstance(outputs, list) and len(outputs) == 2: 

            if self._output_graph_wrapper.adjacency != outputs[0]: 
                self._output_graph_wrapper.adjacency = outputs[0]

            if self._output_graph_wrapper.nodes != outputs[1]: 
                self._output_graph_wrapper.adjacency = outputs[1]

        return self._output_graph_wrapper
        