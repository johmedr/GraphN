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
                 basis=-1, 
                 activation=None,
                 use_attention=False,
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
        self.basis = basis
        self.activation = activations.get(activation)
        self.use_attention = use_attention
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
        adj_shape = to_list(input_shape.adjacency_shape)

        assert 4 > len(x_shape) >= 2, "Expected at least than 2 dims, get %s" % (
            x_shape,)

        for a in adj_shape:
            assert 4 > len(a) >= 2, "Expected at least than 2 dims, get %s" % (a,)

        if self.basis != -1: 
            self.basis = min(self.basis, len(adj_shape))
        else: 
            self.basis = len(adj_shape)

        self.kernel = [self.add_weight(
            shape=(x_shape[-1], self.filters),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        ) for _ in range(self.basis)]

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

    def call(self, inputs):
        """ 
        x: List/GraphWrapper 
          - the adjacency matrix (shape: (n_nodes, n_nodes))
          - the nodes features (shape: (n_nodes, n_features))
          TODO: implement here the renormalization trick """
        if isinstance(inputs, GraphWrapper):
            nodes = inputs.nodes
            adjacency = inputs.adjacency
        else:
            raise ValueError()

        adj_ls = to_list(adjacency)

        supports = []
        for k in self.kernel:
            supports.append(K.dot(nodes, k))

        features = []
        for x, a in zip(supports, adj_ls):
            if len(K.int_shape(x)) == 3:
                x = K.permute_dimensions(x, (1,0,2))
                x = K.batch_dot(a, x)
                x = K.reshape(x, [-1, inputs.n_nodes, self.filters])
            else:
                x = K.dot(a, x)
            features.append(x)

        if len(adj_ls) > 1: 
            features = [K.expand_dims(x, axis=0) for x in features]

        if self.use_attention: 
            raise NotImplementedError()
        elif len(features) > 1: 
            features = K.concatenate(features, axis=0)
            features = K.sum(features, axis=0)
        else:
            features = features[0]

        if self.use_bias:
            features = K.bias_add(features, self.bias,
                                   data_format='channels_last')
        
        if self.activation is not None:
            features = self.activation(features)

        return self.make_output_graph(adjacency=adjacency, nodes=features)

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
