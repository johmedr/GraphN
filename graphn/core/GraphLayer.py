from keras.engine.topology import Layer

class GraphLayer(Layer):
	def __init__(self, **kwargs):
		super(GraphLayer, self).__init__(**kwargs)
		self._output_graph_wrapper = None

	def __call__(self, inputs, **kwargs):
		outputs = super(GraphLayer, self).__call__(inputs, **kwargs)

		if isinstance(outputs, list) and len(outputs) == 2:
			outputs = self._output_graph_wrapper
			
		return outputs
