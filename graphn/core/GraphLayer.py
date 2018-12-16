from keras.engine.topology import Layer
from itertools import chain

from .GraphWrapper import GraphWrapper


class GraphLayer(Layer):
    """ 
    An override of keras.engine.topology.Layer working on graphs. Only two differences 
    when writting a class inheriting graphn.GraphLayer: 

    1. At the end of the usual 'call()' method, call the method 'self.make_output_graph()', 
    passing the (new) adjacency and the (new) nodes as arguments. This function will return 
    a GraphWrapper object that can be returned from the 'call()' method. 

    2. If you called the aforementionned 'make_output_graph()' method, you don't need to 
    override the 'compute_output_shape()', it will be deduced from the output graph's shape.   
    """

    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)
        self._output_graph_wrapper = GraphWrapper(
            name="graph_wrapper_%s" % self.name)

    def make_output_graph(self, adjacency, nodes):
        """ 
        Constructs the 'GraphWrapper' output using the given adjacency and nodes 
        and returns it. 
        Shapes of adjacency and nodes will be checked.
        Args: 
        - adjacency: (..., N, N) tensor 
        - nodes: (..., N, F) tensor
        where N is the number of nodes and F the number of features. 
        """
        if not self._built:
            raise ValueError("The layer is not built yet. \
                make_output_graph() should be called in the call() method.")

        self._output_graph_wrapper.build(adjacency=adjacency, nodes=nodes)
        return self._output_graph_wrapper

    def compute_output_shape(self, input_shape):
        return self._output_graph_wrapper.shape

    def __call__(self, inputs, **kwargs):
        """
        inputs: a GraphWrapper object or 
        a list of 'adjacency, nodes'. 
        """

        # Keras do not supports nested lists, unpack nodes and adjacency
        if isinstance(inputs, list) and not isinstance(inputs, GraphWrapper):
            _inputs = list(chain.from_iterable(
                [i for i in inputs if isinstance(i, GraphWrapper)]))
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
