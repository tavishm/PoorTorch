## Torch clone library
import random
import math

class poortorch():

    ## Tensor Creation Functions
    def rand(shape: list, v_range=(0,1), requires_grad=False):
        l = poortorch.tensor._create_list_iterable(shape, random.uniform, v_range[0], v_range[1], dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)

    def randn(shape: list, requires_grad=False, mean=0, std=1):
        l = poortorch.tensor._create_list_iterable(shape, random.gauss, mean, std, dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)
    
    def zeros(shape: list, requires_grad=False):
        l = poortorch.tensor._create_list_iterable(shape, lambda: 0, dim=0)
        return poortorch.tensor(l, requires_grad=requires_grad)
    
    ## Tensor Editing Functions
    def exp(xt: 'poortorch.tensor'):
        l = poortorch.tensor._edit_list_iterable(xt.__data__, math.exp)
        return poortorch.tensor(l, history=[xt], operator='exp')
    
    ## Custom functions
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
        

    class tensor():
        def __init__(self, x: list, history: list =[], operator: str =None, requires_grad: bool = False):
            self.__data__ = x
            self.shape = self._shape()
            self.requires_grad = False # This seems to be overridden by the parameter, consider removing one
            self.history = history
            self.operator = operator
            self.requires_grad = requires_grad

        def __str__(self):
            return f"PoorTorch({str(self.__data__)})"

        def __add__(self, yt: 'poortorch.tensor'): 
            if self.shape != yt.shape:
                raise Exception("Given two tensors don't have the same shape 😔")
            
            return poortorch.tensor(poortorch.tensor._add_iterable(self.__data__, yt.__data__), history=[self, yt], operator='+')
        

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
            return tuple(poortorch.tensor._shape_iterable(xl_data)[::-1])
        
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
        def _edit_list_iterable(xl: list, fx, *args, **kwargs):
            if isinstance(xl, (int, float)):
                return fx(xl, *args, **kwargs)
            else:
                out = []
                for xi in xl:
                    out.append(poortorch.tensor._edit_list_iterable(xi, fx, *args, **kwargs))
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
        def _shape_iterable(xl: list):
            if not isinstance(xl, list):
                raise Exception("Given list does not have a definite shape 😔")

            elif all(isinstance(i, (int, float)) for i in xl):
                return [len(xl)]
            
            elif not all(isinstance(i, (int, float, list)) for i in xl):
                raise Exception("Given list has elements other than list, int or float 😔")
            
            else:
                shape = []
                ds = []
                for k_item in xl: # Changed from iterating by index to iterating by item
                    ds.append(poortorch.tensor._shape_iterable(k_item))

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

k = poortorch.tensor([[1,2], [3,4]])
k = poortorch.transpose(k, 0, 1)

i = poortorch.tensor([[1,0], [0,1]])
print(k@i)