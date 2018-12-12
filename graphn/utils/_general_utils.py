
def to_list(x):     
    if isinstance(x, list):
        return x
    else: 
        return [x]

def to_tuple(x):     
    if isinstance(x, tuple):
        return x
    else: 
        return (x,)