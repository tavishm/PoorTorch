## Torch clone library
import random
import math

class poortorch():

    ## Tensor Creation Functions
    @staticmethod
    def rand(shape: tuple, v_range=(0,1), requires_grad=False):
        l = poortorch.tensor._create_list_iterable(list(shape), random.uniform, v_range[0], v_range[1], dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)

    @staticmethod
    def randn(shape: tuple, requires_grad=False, mean=0, std=1):
        l = poortorch.tensor._create_list_iterable(list(shape), random.gauss, mean, std, dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)
    
    @staticmethod
    def zeros(shape: tuple, requires_grad=False): # TODO: Update this to tuple
        l = poortorch.tensor._create_list_iterable(list(shape), lambda: 0, dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)
    
    @staticmethod
    def full(x: int | float , shape: tuple, requires_grad=False): # TODO: Update this to tuple
        l = poortorch.tensor._create_list_iterable(list(shape), lambda: x, dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)
    
    ## Tensor Editing Functions
    @staticmethod
    def exp(xt: 'poortorch.tensor'): 
        l = poortorch.tensor._edit_list_iterable(xt.__data__, math.exp)
        return poortorch.tensor(l, history=[xt], operator='exp')
    
    @staticmethod
    def log(xt: 'poortorch.tensor'):
        l = poortorch.tensor._edit_list_iterable(xt.__data__, math.log, condition=lambda x: x > 0) # Logarithm is only defined for positive numbers
        return poortorch.tensor(l, history=[xt], operator='log')
    
    # TODO: Check for numerical stability
    @staticmethod
    def tanh(xt: 'poortorch.tensor'):
        l = poortorch.tensor._edit_list_iterable(xt.__data__, math.tanh)
        return poortorch.tensor(l, history=[xt], operator='tanh')
    
    ## Tensor to Number functions
    @staticmethod
    def mean(xt: 'poortorch.tensor', dim: int =None): # TODO: Make this work with n-axes tensors
        if xt.dtype != 'float':
            raise Exception(f"Can only calculate mean for float tensors 😔")
        
        if dim==None:
            total = 0
            numvals = 0
            def _mean_iterable(xl):
                nonlocal total, numvals
                if isinstance(xl, (int, float)):
                    total+=xl
                    numvals+=1
                else:
                    for xi in xl:
                        _mean_iterable(xi)
            _mean_iterable(xt.__data__)        
            return poortorch.tensor(total/numvals, history=[xt], operator='mean-dim-None')
        
        else:
            def _mean_along_axis(data, axis):
                # Base‐case: axis 0
                if axis == 0:
                    # If data elements are themselves lists, zip and recurse
                    if isinstance(data[0], list):
                        length = len(data)
                        result = []
                        for group in zip(*data):  # group is a tuple of length `length`
                            # group[0] might be scalar or list
                            if isinstance(group[0], list):
                                # Recurse to average each sub‐list
                                result.append(_mean_along_axis(list(group), 0))
                            else:
                                # Direct scalar mean
                                result.append(sum(group) / length)
                        return result
                    else:
                        # data is a flat list of scalars → return a single scalar mean
                        return sum(data) / len(data)

                # Recursive descent: reduce axis deeper in each sub‐list
                return [_mean_along_axis(sub, axis - 1) for sub in data]
                        
            return poortorch.tensor(_mean_along_axis(xt.__data__, dim), history=[xt], operator=f'mean-dim-{dim}')
                    
    ## Custom functions
    @staticmethod
    def transpose(xt: 'poortorch.tensor', dim0: int, dim1: int):
        new_shape = list(xt.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        bl = poortorch.zeros(new_shape)
        l = poortorch._transpose_iterable(xt.__data__, bl.__data__, dim0, dim1)
        return poortorch.tensor(l, history=[xt], operator=f'transpose-{dim0}-{dim1}')

    @staticmethod
    def _transpose_iterable(xl: list, yl: list, dim0: int, dim1: int, pos: list =None):
        if isinstance(xl, (int, float)):
            target_pos = pos.copy() 
            target_pos[dim0], target_pos[dim1] = target_pos[dim1], target_pos[dim0]
            return poortorch.tensor._variable_swap(yl, target_pos, xl)
        else:
            if pos is None: pos = []
            for i, x_item in enumerate(xl):
                pos.append(i)
                yl = poortorch._transpose_iterable(x_item, yl, dim0, dim1, pos)
                pos.pop() 
            return yl 
    
    class nn():

        # TODO: Check for numerical stability
        @staticmethod
        def sigmoid(xt: 'poortorch.tensor'):
            l = poortorch.tensor._edit_list_iterable(xt.__data__, lambda x: 1 / (1 + math.exp(-x)))
            return poortorch.tensor(l, history=[xt], operator='sigmoid')
        
        # TODO: Check for numerical stability
        @staticmethod
        def ReLU(xt: 'poortorch.tensor'):
            l = poortorch.tensor._edit_list_iterable(xt.__data__, lambda x: max(0, x))
            return poortorch.tensor(l, history=[xt], operator='ReLU')

    class tensor():
        def __init__(self, x: list | int | float, history: list =[], operator: str =None, requires_grad: bool = False):
            if isinstance(x, (float, int)): 
                self.__data__ = [x]
                self.info_type = "value"
            else:
                self.__data__ = x
                self.info_type = "tensor"

            self.dtype = "int" # TODO: Replace with class for int and float
            self.shape = self._shape()
            self.history = history
            self.operator = operator
            self.requires_grad = requires_grad
            self.grad = None

        def get(self, *idx):
            shape = poortorch.tensor._shape(self)
            if len(idx)!= len(shape):
                raise Exception(f"{len(shape)} indices required 😔")
            elif not all(isinstance(i,int) for i in idx):
                raise Exception('indices must be integers 😔')
            elif not all(i[0]<i[1] for i in zip(idx, shape)):
                raise Exception(f"index out of range 😔\nShape of tensor: {shape}")
            else:
                def _get(xl: list, idx: list):
                    a = xl[idx[0]]
                    if isinstance(a, (int, float)):
                        return a
                    else:
                        return _get(a,idx[1:])
                return _get(self.__data__, idx)
    
        def backward(self):
            if self.operator == '+':           
                def _backward(self):
                    # z = x + y
                    # dz / dx = 1; dz / dy = 1
                    # Gradients flow back equally in such simple element-wise addition operations. 
                    if self.grad == None: # To check for chain rule predecessors
                        print(list(self.history[0].shape), poortorch.zeros(self.history[0].shape))
                        self.history[0].grad = poortorch.full(1, self.history[0].shape)
                        self.history[1].grad = poortorch.full(1, self.history[1].shape)
                    else: 
                        self.history[0].grad = self.grad * poortorch.full(1, self.history[0].shape)
                        self.history[1].grad = self.grad * poortorch.full(1, self.history[1].shape)

                    # Calculating gradients for the tensors that zt was made of
                    self.history[0].backward()
                    self.history[1].backward()

                _backward(self)
            
            if self.operator == '-':           
                def _backward(self):
                    # z = x + y
                    # dz / dx = 1; dz / dy = -1
                    if self.grad == None: # To check for chain rule predecessors
                        print(list(self.history[0].shape), poortorch.zeros(self.history[0].shape))
                        self.history[0].grad = poortorch.full(1, self.history[0].shape)
                        self.history[1].grad = poortorch.full(1, self.history[1].shape)
                    else: 
                        self.history[0].grad = self.grad * poortorch.full(1, self.history[0].shape)
                        self.history[1].grad = self.grad * poortorch.full(-1, self.history[1].shape)

                    # Calculating gradients for the tensors that zt was made of
                    self.history[0].backward()
                    self.history[1].backward()

                _backward(self)

            elif self.operator == 'mean':
                def _backward(self):
                    # y = x1 + x2 + x3 + ... + xn/ (n)
                    # dy / dxi = 1/n

                    if len(self.history[0].shape) > 1: raise Exception("Backward for mean of 2 axes tensors has not been implemented yet 😔")

                    if self.grad == None:
                        self.history[0].grad = poortorch.tensor([1/self.history[0].shape[0]] * self.history[0].shape[0])
                    else: 
                        self.history[0].grad = self.grad[0] * poortorch.tensor([1/self.history[0].shape[0]] * self.history[0].shape[0])

                    self.history[0].backward()
                    
                _backward(self)

            # Removing grad for tensors that don't need to store grad
            for t in self.history:
                if not t.requires_grad: t.grad = None

        def __getitem__(self, idx: int):
            if isinstance(idx, int):
                if not isinstance(idx, int):
                    raise Exception('index must be an integer 😔')
                elif idx>= len(self.__data__) or idx< -len(self.__data__):
                    raise Exception('index out of range 😔')
                else:
                    return self.__data__[idx]
            else:
                l = [idx.start,idx.stop, idx.step]
                if not all( isinstance(i, (int)) or i==None for i in l):
                    raise Exception('indices must be an integers 😔')
                if not all(i< len(self.__data__) or i>= -len(self.__data__) for i in l if i!= None):
                    raise Exception('index out of range 😔')
                else: 
                    return self.__data__[l[0]:l[1]:l[2]]
                      

        def __str__(self):
            return f"PoorTorch({str(self.__data__)})"
        
        def __int__(self):
            if self.info_type != "value":
                raise Exception("Cannot convert tensor to int 😔")
            return int(self.__data__[0])
        
        def __float__(self):
            if self.info_type != "value":
                raise Exception("Cannot convert tensor to float 😔")
            return float(self.__data__[0])
        
        def __iter__(self):
            if self.info_type != "tensor":
                raise Exception("Cannot convert value to list 😔")
            return iter(self.__data__) 

        def __add__(self, yt: 'poortorch.tensor'): 
            if self.shape != yt.shape:
                raise Exception("Given two tensors don't have the same shape 😔")
            return poortorch.tensor(poortorch.tensor._add_iterable(self.__data__, yt.__data__), history=[self, yt], operator='+')
        
        def __sub__(self, yt: 'poortorch.tensor'): 
            if self.shape != yt.shape:
                raise Exception("Given two tensors don't have the same shape 😔")
            
            return poortorch.tensor(poortorch.tensor._sub_iterable(self.__data__, yt.__data__), history=[self, yt], operator='-')        

        def __mul__(self, yt: 'poortorch.tensor'): 
            if self.shape != yt.shape:
                raise Exception("Given two tensors don't have the same shape 😔")
        
            return poortorch.tensor(poortorch.tensor._mul_iterable(self.__data__, yt.__data__), history=[self, yt], operator='*')

        def __matmul__(self, yt: 'poortorch.tensor'):
            if len(self.shape) != 2 or len(yt.shape) != 2:
                raise Exception("Matrix multiplication is only supported for 2D tensors 😔")
            
            if self.shape[1] != yt.shape[0]:
                raise Exception(f"Cannot multiply tensors with shapes {self.shape} and {yt.shape} 😔")

            rows_self = self.shape[0]
            cols_self = self.shape[1] # == rows_yt
            cols_yt = yt.shape[0] if len(yt.shape) == 1 else yt.shape[1]


            result_data = [[0 for _ in range(cols_yt)] for _ in range(rows_self)]

            for i in range(rows_self):
                for j in range(cols_yt):
                    sum_val = 0
                    for k_idx in range(cols_self): # k_idx is the shared dimension
                        sum_val += self.__data__[i][k_idx] * yt.__data__[k_idx][j]
                        result_data[i][j] = sum_val
                
            return poortorch.tensor(result_data, history=[self, yt], operator='@')


        def _shape(self):
            xl_data = self.__data__ # Renamed from i_list as it's internal data, not a direct parameter
            return tuple(poortorch.tensor._shape_iterable(xl_data, self)[::-1])
        
        # Tensor Editing Operations
        def float(self):
            l = poortorch.tensor._edit_list_iterable(self.__data__, float)
            return poortorch.tensor(l, history=[self], operator='float')
    
        @staticmethod
        def _create_list_iterable(shape: list, fx, *args, dim: int =0, **kwargs):
            if dim+1 == len(shape):
                out = []
                for _ in range(shape[-1]):
                    out.append(fx(*args, **kwargs))
                return out
            else: 
                out = []
                for _ in range(shape[dim]): # Renamed x to _ as it's not used
                    out.append(poortorch.tensor._create_list_iterable(shape, fx, *args, dim=dim+1, **kwargs))
                return out

        @staticmethod
        def _edit_list_iterable(xl: list, fx, *args, condition=None, **kwargs):
            if isinstance(xl, (int, float)):
                if condition is None: return fx(xl, *args, **kwargs)
                elif condition(xl):
                    return fx(xl, *args, **kwargs)
                else:
                    return float('nan')
            else:
                out = []
                for xi in xl:
                    out.append(poortorch.tensor._edit_list_iterable(xi, fx, condition=condition,*args, **kwargs))
                return out    


        @staticmethod
        def _add_iterable(xl: list, yl: list):
            if isinstance(xl, (int, float)):
                return xl+yl
            else:
                out = []
                for xi, yi in zip(xl, yl):
                    out.append(poortorch.tensor._add_iterable(xi, yi))
                return out

        @staticmethod
        def _mul_iterable(xl: list, yl: list):
            if isinstance(xl, (int, float)):
                return xl*yl
            else:
                out = []
                for xi, yi in zip(xl, yl):
                    out.append(poortorch.tensor._mul_iterable(xi, yi))
                return out


        @staticmethod
        def _shape_iterable(xl: list, self):
            if not isinstance(xl, list):
                raise Exception("Given list does not have a definite shape 😔")

            elif all(isinstance(i, (int, float)) for i in xl):
                for i in xl:
                    if isinstance(i, float):
                        self.dtype = "float"
                
                return [len(xl)]
            
            elif not all(isinstance(i, (int, float, list)) for i in xl):
                raise Exception("Given list has elements other than list, int or float 😔")
            
            else:
                shape = []
                ds = []
                for k_item in xl: # Changed from iterating by index to iterating by item
                    ds.append(poortorch.tensor._shape_iterable(k_item, self))

                same_shape = all(ds[0] == j for j in ds)
                if not same_shape:
                    raise Exception("Given list does not have a definite shape 😔")
                if same_shape:
                    shape.extend(ds[0])
                    shape.append(len(xl))

            return shape     
            
        @staticmethod
        def _variable_splice(xl: list, idx: list):
            if len(idx) == 1: idx = idx[0]
            if isinstance(idx, int): return xl[idx]
            else:
                next_depth_list = xl[idx[0]]
                return poortorch.tensor._variable_splice(next_depth_list, idx[1:])

        @staticmethod
        def _variable_swap(xl: list, idx: list, value):
            if len(idx) == 1: idx = idx[0]
            print(xl, idx)
            if isinstance(idx, (int, float)): 
                xl[idx] = value
                return xl
            else:
                next_depth_list = xl[idx[0]]
                xl[idx[0]] = poortorch.tensor._variable_swap(next_depth_list, idx[1:], value)
                return xl

a = poortorch.tensor([1,2,3], requires_grad=True).float()
b = poortorch.tensor([0,1,2], requires_grad=True).float()
c = a + b
d = poortorch.mean(c)

d.backward()