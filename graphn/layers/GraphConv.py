import keras.backend as K

from keras import activations
from keras import regularizers
from keras import initializers
from keras import constraints

from keras.utils.generic_utils import to_list

from ..core import GraphLayer
from ..core import GraphShape
from ..core import GraphWrapper


class GraphConv(GraphLayer):
    """
    A graph convolution layer using the first-order approximation of 
    Chebyshev polynomials, as suggested in "Semi-supervised classification with Graph Convolutional 
    Networks" (T. Kipf and M. Welling, 2016, https://arxiv.org/pdf/1609.02907.pdf)
    """

    def __init__(self, filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphConv, self).__init__(**kwargs)

        self.filters = filters
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        print(input_shape)
        assert isinstance(input_shape, GraphShape)

        x_shape = input_shape.nodes_shape
        adj_shape = input_shape.adjacency_shape

        assert len(x_shape) >= 2, "Expected at least than 2 dims, get %s" % (
            x_shape,)

        for a in to_list(adj_shape): 
            assert len(a) >= 2, "Expected at least than 2 dims, get %s" % (a,)

        self.kernel = self.add_weight(
            shape=(x_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

        self.built = True

    def call(self, inputs):
        """ 
        x: List/GraphWrapper 
          - the adjacency matrix (shape: (n_nodes, n_nodes))
          - the nodes features (shape: (n_nodes, n_features))
          TODO: implement here the renormalization trick """
        assert isinstance(inputs, GraphWrapper)
        nodes, adjacency = inputs

        new_nodes = K.dot(nodes, self.kernel)

        if len(K.int_shape(nodes)) > 2:
            new_nodes = K.batch_dot(adjacency, new_nodes)
        else:
            new_nodes = K.dot(adjacency, new_nodes)

        if self.use_bias:
            new_nodes = K.bias_add(new_nodes, self.bias,
                                   data_format='channels_last')
        if self.activation is not None:
            new_nodes = self.activation(new_nodes)

        return self.make_output_graph(adjacency=adjacency, nodes=new_nodes)

    def get_config(self):
        config = {
            'filters': self.filters,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
