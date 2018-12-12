
def to_list(x):     
    if isinstance(x, list):
        return x
    if isinstance(x, tuple): 
    	return list(x)
    else: 
        return [x]

def to_tuple(x):     
    if isinstance(x, tuple):
        return x
    if isinstance(x, list): 
    	return tuple(x)
    else: 
        return (x,)