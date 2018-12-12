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
			assert(isinstance(inputs, list))

			A, x = inputs

			self.n_nodes = K.int_shape(A)[-1]
			self.n_features = K.int_shape(x)[-1]

			self.build(inputs)

		else: 
			raise AttributeError("You must provide either inputs or shapes to build the GraphWrapper.")

	def build(self, inputs): 
		assert(isinstance(inputs, list))
		A, x = inputs

		assert K.int_shape(A)[-1] ==  K.int_shape(A)[-2] == K.int_shape(x)[-2] == self.n_nodes, \
			"Wrong shapes A: %s, x: %s for %d nodes."%(K.int_shape(A)[-2:], K.int_shape(x)[-2], self.n_nodes)
		assert K.int_shape(x)[-1] == self.n_features

		self._build(A, x)

		self._built = True

	def _build(self, A, x): 
		self.adjacency = A
		self.nodes = x

		self._keras_shape = [K.int_shape(A), K.int_shape(x)]

		super(GraphWrapper, self).clear()
		super(GraphWrapper, self).extend([self.adjacency, self.nodes])

	def get_shape(self):
		assert(self._built)

		return self._keras_shape

	shape = property(get_shape)