class _Wrapper(list): 
    def __init__(self, **kwargs): 
        super(_Wrapper, self).__init__(self, **kwargs)

    # Private methods for childs
    def _extend(self, arg): 
        super(_Wrapper, self).extend(arg)

    def _clear(self): 
        super(_Wrapper, self).clear()

    @staticmethod
    def _disabled_method(): 
        raise NotImplementedError("This method is not available on this object.")

    # GraphWrapper inherits from a list for Keras compatibility, but 
    # override list default methods to ensure avoid a weird behavior
    def append(self, **kwargs): _Wrapper._disabled_method()
    def extend(self, **kwargs): _Wrapper._disabled_method()
    def clear(self, **kwargs): _Wrapper._disabled_method()
    def insert(self, **kwargs): _Wrapper._disabled_method()
    def reverse(self, **kwargs): _Wrapper._disabled_method()
    def remove(self, **kwargs): _Wrapper._disabled_method()
    def sort(self, **kwargs): _Wrapper._disabled_method()
    def pop(self, **kwargs): _Wrapper._disabled_method()
    def __set_items__(self, **kwargs): _Wrapper._disabled_method()