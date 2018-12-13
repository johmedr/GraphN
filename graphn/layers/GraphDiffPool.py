import keras.backend as K

from .GraphPoolingCell import GraphPoolingCell
from .GraphConv import GraphConv
from ..core import GraphLayer, GraphWrapper

class GraphDiffPool(GraphLayer):
	def __init__(self, **kwargs):
		pass