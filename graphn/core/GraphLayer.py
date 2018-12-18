from keras.engine.topology import Layer
from keras.engine.base_layer import _collect_previous_mask
from keras.engine.base_layer import _collect_input_shape

import keras.backend as K

from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras.utils.generic_utils import is_all_none
from keras.utils.generic_utils import has_arg

import copy

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

    For the moment, handles only one ouput graph (to be implemented).
    """

    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)
        self._output_graph_wrapper = GraphWrapper(
            name="graph_wrapper_%s" % self.name)

    def make_output_graph(self, nodes, adjacency):
        """ 
        Constructs the 'GraphWrapper' output using the given adjacency and nodes 
        and returns it. 
        Shapes of adjacency and nodes will be checked.
        Args: 
        - nodes: (..., N, F) tensor
        - adjacency: (..., N, N) tensor 
        where N is the number of nodes and F the number of features. 
        """
        if not self._built:
            raise ValueError("The layer is not built yet. \
                make_output_graph() should be called in the call() method.")

        self._output_graph_wrapper.build(nodes=nodes, adjacency=adjacency)
        return self._output_graph_wrapper

    def compute_output_shape(self, input_shape):
        return self._output_graph_wrapper.shape

    def __call__(self, inputs, **kwargs):
        """
        Overriding the Layer's __call__ method for graph wrappers. 
        The overriding is mostly based on the __call__ code from 
        'keras/engine/base_layer.py' for the moment, with some changes 
        to handle graphs. 
        """
        # If arguments are 'keras-style' arguments, let keras to the job
        if K.is_tensor(inputs) or (isinstance(inputs, list) and not isinstance(inputs, GraphWrapper)):
            output = super(GraphLayer, self).__call__(inputs, **kwargs)
        else:
            if isinstance(inputs, list) and not isinstance(inputs, GraphWrapper):
                inputs = inputs[:]

            with K.name_scope(self.name):
                # Build the layer
                if not self.built:
                    # Check the comatibilty of inputs for each inputs (some can
                    # be GraphWrapper)
                    if isinstance(inputs, list) and not isinstance(inputs, GraphWrapper):
                        for i in inputs:
                            self.assert_input_compatibility(i)

                    # Collect input shapes to build layer.
                    input_shapes = []

                    if isinstance(inputs, GraphWrapper):
                        inputs = [inputs]

                    for x_elem in inputs:
                        if hasattr(x_elem, '_keras_shape'):
                            # For a GraphWrapper, _keras_shape is a GraphShape
                            # object
                            input_shapes.append(x_elem._keras_shape)
                        elif hasattr(K, 'int_shape'):
                            input_shapes.append(K.int_shape(x_elem))
                        else:
                            raise ValueError('You tried to call layer "' +
                                             self.name +
                                             '". This layer has no information'
                                             ' about its expected input shape, '
                                             'and thus cannot be built. '
                                             'You can build it manually via: '
                                             '`layer.build(batch_input_shape)`')

                    self.build(unpack_singleton(input_shapes))
                    self._built = True

                    # Load weights that were specified at layer instantiation.
                    if self._initial_weights is not None:
                        self.set_weights(self._initial_weights)

                # Raise exceptions in case the input is not compatible
                # with the input_spec set at build time.
                if isinstance(inputs, list) and not isinstance(inputs, GraphWrapper):
                    for i in inputs:
                        self.assert_input_compatibility(i)

                # Handle mask propagation.
                previous_mask = _collect_previous_mask(inputs)
                user_kwargs = copy.copy(kwargs)
                if not is_all_none(previous_mask):
                    # The previous layer generated a mask.
                    if has_arg(self.call, 'mask'):
                        if 'mask' not in kwargs:
                            # If mask is explicitly passed to __call__,
                            # we should override the default mask.
                            kwargs['mask'] = previous_mask
                # Handle automatic shape inference (only useful for Theano).
                input_shape = _collect_input_shape(inputs)

                # Actually call the layer,
                # collecting output(s), mask(s), and shape(s).
                # Note that inpputs can hold graph wrappers now
                output = self.call(unpack_singleton(inputs), **kwargs)
                output_mask = self.compute_mask(inputs, previous_mask)

                # If the layer returns tensors from its inputs, unmodified,
                # we copy them to avoid loss of tensor metadata.
                # output_ls = to_list(output)
                if isinstance(output, GraphWrapper):
                    output_ls = [output]
                # Unpack wrappers
                inputs_ls = list(chain.from_iterable(
                    [i for i in inputs if isinstance(i, GraphWrapper)]))
                inputs_ls += [
                    i for i in inputs if not isinstance(i, GraphWrapper)]

                # Unpack adjacency and nodes
                inpadj_ls = list(chain.from_iterable(
                    [to_list(i.adjacency) for i in inputs if isinstance(i, GraphWrapper)]))
                inpnod_ls = to_list(
                    [i.nodes for i in inputs if isinstance(i, GraphWrapper)])

                output_ls_copy = []
                adj_ls = []
                for x in output_ls:
                    if K.is_tensor(x) and x in inputs_ls:
                        x = K.identity(x)
                    # Apply adjacency-wise and node-wise identity
                    elif isinstance(x, GraphWrapper):
                        # Store changed or copy of unchanged adjacency matrices
                        adj_ls.clear()
                        for adj in to_list(x.adjacency):
                            if adj in inpadj_ls:
                                adj_ls.append(K.identity(adj))
                            else:
                                adj_ls.append(adj)
                        # Assign to output graph
                        x.adjacency = unpack_singleton(adj_ls)
                        # Store unchanged nodes
                        if x.nodes in inpnod_ls:
                            x.nodes = K.identity(x.nodes)
                    output_ls_copy.append(x)

                output = unpack_singleton(output_ls_copy)

                # Inferring the output shape is only relevant for Theano.
                if all([s is not None
                        for s in to_list(input_shape)]):
                    output_shape = self.compute_output_shape(input_shape)
                else:
                    if isinstance(input_shape, list):
                        output_shape = [None for _ in input_shape]
                    else:
                        output_shape = None

                if (not isinstance(output_mask, (list, tuple)) and
                        len(output_ls) > 1):
                    # Augment the mask to match the length of the output.
                    output_mask = [output_mask] * len(output_ls)

                # Add an inbound node to the layer, so that it keeps track
                # of the call and of all new variables created during the call.
                # This also updates the layer history of the output tensor(s).
                # If the input tensor(s) had not previous Keras history,
                # this does nothing.
                self._add_inbound_node(input_tensors=inputs_ls,
                                       output_tensors=unpack_singleton(output),
                                       input_masks=previous_mask,
                                       output_masks=output_mask,
                                       input_shapes=input_shape,
                                       output_shapes=output_shape,
                                       arguments=user_kwargs)

                # Apply activity regularizer if any:
                if (hasattr(self, 'activity_regularizer') and
                        self.activity_regularizer is not None):
                    with K.name_scope('activity_regularizer'):
                        regularization_losses = [
                            self.activity_regularizer(x)
                            for x in to_list(output)]
                    self.add_loss(regularization_losses,
                                  inputs=to_list(inputs))
        return output
