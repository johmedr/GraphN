import keras.backend as K 

def frobenius_norm(x): 
	return K.sqrt(K.sum(K.square(x)))

def entropy(x, axis=None): 
	return K.mean(x * K.log(x), axis=None)