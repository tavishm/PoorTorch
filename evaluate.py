# -*- coding: utf-8 -*-
from main import poortorch
import math
import sys



class EvaluatePoortorch():
    def __init__(self, torch_eval=True):
        self.functions_to_evaluate = {

            # Properly checkable functions
            poortorch.rand: self.rand_eval,
            poortorch.randn: self.randn_eval,
            poortorch.zeros: self.zeros_eval,
            poortorch.exp: self.exp_eval,
            poortorch.log: self.log_eval,
            poortorch.tanh: self.tanh_eval,
            poortorch.transpose: self.transpose_eval,
            poortorch.nn.sigmoid: self.sigmoid_eval,
            poortorch.nn.ReLU: self.ReLU_eval,
            
            # New evaluation functions for tensor operations
            '__str__': self.str_eval,
            '__int__': self.int_eval,
            '__float__': self.float_eval,
            '__iter__': self.iter_eval,
            '__add__': self.add_eval,
            '__mul__': self.mul_eval,
            '__matmul__': self.matmul_eval,
            'float': self.float_method_eval,

        }
        self.torch_eval = torch_eval
        if torch_eval:
            self.functions_to_evaluate.update({
                poortorch.mean: self.mean_eval,
            })

    def eval(self):
        for poortorch_f, eval_f in self.functions_to_evaluate.items():
            passed = eval_f()
            # Handle special case for tensor methods that don't have __name__
            if isinstance(poortorch_f, str):
                print(f"{poortorch_f} evaluation result: {'✅' if passed else '❌'}")
            else:
                print(f"{poortorch_f.__name__} evaluation result: {'✅' if passed else '❌'}")

    @staticmethod
    def _check_condition(xl: list, condition_f):
        if isinstance(xl, (int, float)):
            return condition_f(xl)
        else:
            truth_list = []
            for xi in xl:
                truth_list.append(EvaluatePoortorch._check_condition(xi, condition_f))
            return all(truth_list)

    @staticmethod
    def _check_operation_condition(xl_o: list, xl_f: list, operation_f):
        if isinstance(xl_o, (int, float)) and isinstance(xl_f, (int, float)):
            return operation_f(xl_o, xl_f)
        else:
            truth_list = []
            for xi_o, xi_f in zip(xl_o, xl_f):
                truth_list.append(EvaluatePoortorch._check_operation_condition(xi_o, xi_f, operation_f))
            return all(truth_list)

    def rand_eval(self):
        test_tensor = poortorch.rand((1,2,3,4), v_range=(7,8), requires_grad=True)
        in_range = EvaluatePoortorch._check_condition(test_tensor.__data__, lambda k: 7<=k<=8)
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==True and in_range:
            return True
        else:   
            return False
        
    def randn_eval(self): # TODO: Check if standard deviation is correct - 
        # TODO: I have reason to suspect STD is not working because even though 95% times we should have mean in range, it is not the case
        test_tensor = poortorch.randn((10, 20, 30, 4), mean=10, std=0.5, requires_grad=True) # Using huge values to make sure mean converges at expected value
        mean = int(poortorch.mean(test_tensor))
        if test_tensor.shape == (10, 20, 30, 4) and test_tensor.requires_grad == True and 9 < mean and mean < 11: # 95% values will fall here for mean 10 and STD 0.5
            return True
        else:
            return False
        
    def zeros_eval(self):
        test_tensor = poortorch.zeros((1,2,3,4), requires_grad=True)
        in_range = EvaluatePoortorch._check_condition(test_tensor.__data__, lambda k: k==0)
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==True and in_range:
            return True
        else:
            return False
        
    def exp_eval(self):
        xt_o = poortorch.randn((1,2,3,4), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.exp(xt_o)
        in_range = EvaluatePoortorch._check_operation_condition(xt_o.__data__, test_tensor.__data__, lambda x_o, x_f: x_f == math.exp(x_o))
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==False and in_range:
            return True
        else:
            return False
        
    def log_eval(self):
        xt_o = poortorch.randn((1,2,3,4), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.log(xt_o)
        in_range = EvaluatePoortorch._check_operation_condition(xt_o.__data__, test_tensor.__data__, lambda x_o, x_f: x_f == math.log(x_o) if x_o > 0 else math.isnan(x_f)) 
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==False and in_range:
            return True
        else:
            return False
    
    def tanh_eval(self):
        xt_o = poortorch.randn((1,2,3,4), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.tanh(xt_o)
        in_range = EvaluatePoortorch._check_operation_condition(xt_o.__data__, test_tensor.__data__, lambda x_o, x_f: x_f == math.tanh(x_o))
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==False and in_range:
            return True
        else:
            return False
    
    def transpose_eval(self):
        # Test case 1: Simple 2D tensor
        shape1 = (3, 4)
        xt1 = poortorch.randn(shape1, mean=0, std=1, requires_grad=True)
        test_tensor1 = poortorch.transpose(xt1, 0, 1)
        
        # Check shape is transposed correctly
        expected_shape1 = (4, 3)
        shape_correct1 = test_tensor1.shape == expected_shape1
        
        # Manually check a few values to ensure transposition worked
        values_correct1 = True
        for i in range(shape1[0]):
            for j in range(shape1[1]):
                if xt1.__data__[i][j] != test_tensor1.__data__[j][i]:
                    values_correct1 = False
        
        # Test case 2: 3D tensor with different dimension transposition
        shape2 = (2, 3, 4)
        xt2 = poortorch.randn(shape2, mean=0, std=1, requires_grad=True)
        test_tensor2 = poortorch.transpose(xt2, 1, 2)
        
        # Check shape is transposed correctly
        expected_shape2 = (2, 4, 3)
        shape_correct2 = test_tensor2.shape == expected_shape2
        
        # Manually check a few values to ensure transposition worked
        values_correct2 = True
        for i in range(shape2[0]):
            for j in range(shape2[1]):
                for k in range(shape2[2]):
                    if xt2.__data__[i][j][k] != test_tensor2.__data__[i][k][j]:
                        values_correct2 = False
        
        # Test case 3: 3D tensor with outer dimension transposition
        test_tensor3 = poortorch.transpose(xt2, 0, 2)
        
        # Check shape is transposed correctly
        expected_shape3 = (4, 3, 2)
        shape_correct3 = test_tensor3.shape == expected_shape3
        
        # Manually check a few values to ensure transposition worked
        values_correct3 = True
        for i in range(shape2[0]):
            for j in range(shape2[1]):
                for k in range(shape2[2]):
                    if xt2.__data__[i][j][k] != test_tensor3.__data__[k][j][i]:
                        values_correct3 = False
        
        if (shape_correct1 and values_correct1 and 
            shape_correct2 and values_correct2 and
            shape_correct3 and values_correct3):
            return True
        else:
            return False
    
    def sigmoid_eval(self):
        xt_o = poortorch.randn((1, 2, 3, 4), mean=0, std=2, requires_grad=True)
        test_tensor = poortorch.nn.sigmoid(xt_o)
        
        # Check if sigmoid is implemented correctly
        in_range = EvaluatePoortorch._check_operation_condition(
            xt_o.__data__, 
            test_tensor.__data__, 
            lambda x_o, x_f: abs(x_f - (1 / (1 + math.exp(-x_o)))) < 1e-10
        )
        
        # Check bounds: sigmoid should always be between 0 and 1
        bounds_check = EvaluatePoortorch._check_condition(
            test_tensor.__data__,
            lambda x: 0 <= x <= 1
        )
        
        # Check specific known values
        known_values = [
            [-10, 4.539786870243442e-05],
            [-5, 0.0066928509242848554],
            [-1, 0.2689414213699951],
            [0, 0.5],
            [1, 0.7310585786300049],
            [5, 0.9933071490757153],
            [10, 0.9999546021312976]
        ]
        
        known_check = True
        for val, expected in known_values:
            sigmoid_val = 1 / (1 + math.exp(-val))
            if abs(sigmoid_val - expected) > 1e-10:
                known_check = False
        
        if self.torch_eval:
            # Compare with torch implementation
            import torch
            xp_o = torch.tensor(xt_o.__data__, dtype=torch.float32)
            torch_sigmoid = torch.sigmoid(xp_o)
            torch_match = torch.allclose(
                torch_sigmoid, 
                torch.tensor(test_tensor.__data__, dtype=torch.float32),
                rtol=1e-5
            )
            
            if test_tensor.shape == xt_o.shape and test_tensor.requires_grad == False and in_range and bounds_check and known_check and torch_match:
                return True
            else:
                return False
        else:
            if test_tensor.shape == xt_o.shape and test_tensor.requires_grad == False and in_range and bounds_check and known_check:
                return True
            else:
                return False
    
    def ReLU_eval(self):
        xt_o = poortorch.randn((2, 3, 4, 5), mean=0, std=2, requires_grad=True)
        test_tensor = poortorch.nn.ReLU(xt_o)
        
        # Check if ReLU is implemented correctly
        in_range = EvaluatePoortorch._check_operation_condition(
            xt_o.__data__, 
            test_tensor.__data__, 
            lambda x_o, x_f: x_f == max(0, x_o)
        )
        
        # Check bounds: ReLU should always be non-negative
        bounds_check = EvaluatePoortorch._check_condition(
            test_tensor.__data__,
            lambda x: x >= 0
        )
        
        # Check specific known values
        known_values = [
            [-10, 0],
            [-1, 0],
            [0, 0],
            [0.5, 0.5],
            [1, 1],
            [5, 5],
            [10, 10]
        ]
        
        known_check = True
        for val, expected in known_values:
            relu_val = max(0, val)
            if relu_val != expected:
                known_check = False
        
        if self.torch_eval:
            # Compare with torch implementation
            import torch
            xp_o = torch.tensor(xt_o.__data__, dtype=torch.float32)
            torch_relu = torch.relu(xp_o)
            torch_match = torch.allclose(
                torch_relu, 
                torch.tensor(test_tensor.__data__, dtype=torch.float32),
                rtol=1e-5
            )
            
            if test_tensor.shape == xt_o.shape and test_tensor.requires_grad == False and in_range and bounds_check and known_check and torch_match:
                return True
            else:
                return False
        else:
            if test_tensor.shape == xt_o.shape and test_tensor.requires_grad == False and in_range and bounds_check and known_check:
                return True
            else:
                return False
            
    def str_eval(self):
        # Test case 1: Value tensor
        value_tensor = poortorch.tensor(42)
        str_result = str(value_tensor)
        expected_str = "PoorTorch([42])"
        str_correct = str_result == expected_str
        
        # Test case 2: 1D tensor
        tensor_1d = poortorch.tensor([1, 2, 3])
        str_result_1d = str(tensor_1d)
        expected_str_1d = "PoorTorch([1, 2, 3])"
        str_correct_1d = str_result_1d == expected_str_1d
        
        # Test case 3: 2D tensor
        tensor_2d = poortorch.tensor([[1, 2], [3, 4]])
        str_result_2d = str(tensor_2d)
        expected_str_2d = "PoorTorch([[1, 2], [3, 4]])"
        str_correct_2d = str_result_2d == expected_str_2d
        
        if str_correct and str_correct_1d and str_correct_2d:
            return True
        else:
            return False
    
    def int_eval(self):
        # Test case 1: Value tensor with integer
        value_tensor = poortorch.tensor(42)
        int_result = int(value_tensor)
        int_correct = int_result == 42
        
        # Test case 2: Value tensor with float (should truncate)
        float_tensor = poortorch.tensor(42.7)
        int_result_float = int(float_tensor)
        int_correct_float = int_result_float == 42
        
        # Test case 3: Non-value tensor (should raise TypeError)
        tensor_1d = poortorch.tensor([1, 2, 3])
        type_error_raised = False
        try:
            int(tensor_1d)
        except TypeError as e:
            type_error_raised = "Cannot convert tensor to int 😔" in str(e)
        
        if int_correct and int_correct_float and type_error_raised:
            return True
        else:
            return False
    
    def float_eval(self):
        # Test case 1: Value tensor with integer
        value_tensor = poortorch.tensor(42)
        float_result = float(value_tensor)
        float_correct = float_result == 42.0 and isinstance(float_result, float)
        
        # Test case 2: Value tensor with float
        float_tensor = poortorch.tensor(42.7)
        float_result_float = float(float_tensor)
        float_correct_float = float_result_float == 42.7
        
        # Test case 3: Non-value tensor (should raise TypeError)
        tensor_1d = poortorch.tensor([1, 2, 3])
        type_error_raised = False
        try:
            float(tensor_1d)
        except TypeError as e:
            type_error_raised = "Cannot convert tensor to float 😔" in str(e)
        
        if float_correct and float_correct_float and type_error_raised:
            return True
        else:
            return False
    
    def iter_eval(self):
        # Test case 1: 1D tensor
        tensor_1d = poortorch.tensor([1, 2, 3])
        iter_result = list(iter(tensor_1d))
        iter_correct = iter_result == [1, 2, 3]
        
        # Test case 2: 2D tensor
        tensor_2d = poortorch.tensor([[1, 2], [3, 4]])
        iter_result_2d = list(iter(tensor_2d))
        iter_correct_2d = iter_result_2d == [[1, 2], [3, 4]]
        
        # Test case 3: Value tensor (should raise TypeError)
        value_tensor = poortorch.tensor(42)
        type_error_raised = False
        try:
            list(iter(value_tensor))
        except TypeError as e:
            type_error_raised = "Cannot convert value to list 😔" in str(e)
        
        if iter_correct and iter_correct_2d and type_error_raised:
            return True
        else:
            return False

    def add_eval(self):
        # Test case 1: Simple 1D tensors
        a = poortorch.tensor([1, 2, 3])
        b = poortorch.tensor([4, 5, 6])
        result = a + b
        
        # Check correct addition
        expected = [5, 7, 9]
        addition_correct = all(result.__data__[i] == expected[i] for i in range(len(expected)))
        
        # Check shape maintained
        shape_correct = result.shape == (3,)
        
        # Test case 2: 2D tensors
        a_2d = poortorch.tensor([[1, 2], [3, 4]])
        b_2d = poortorch.tensor([[5, 6], [7, 8]])
        result_2d = a_2d + b_2d
        
        # Check correct addition
        expected_2d = [[6, 8], [10, 12]]
        addition_correct_2d = all(
            result_2d.__data__[i][j] == expected_2d[i][j] 
            for i in range(len(expected_2d)) 
            for j in range(len(expected_2d[0]))
        )
        
        # Check shape maintained
        shape_correct_2d = result_2d.shape == (2, 2)
        
        # Test case 3: Mismatched shapes (should raise Exception)
        a_mismatch = poortorch.tensor([1, 2, 3])
        b_mismatch = poortorch.tensor([4, 5])
        error_raised = False
        try:
            a_mismatch + b_mismatch
        except Exception as e:
            error_raised = "don't have the same shape 😔" in str(e)
        
        # Test case 4: Check history tracking for backward propagation
        a_hist = poortorch.tensor([1.0, 2.0], requires_grad=True)
        b_hist = poortorch.tensor([3.0, 4.0], requires_grad=True)
        result_hist = a_hist + b_hist
        
        history_correct = (result_hist.history == [a_hist, b_hist] and 
                           result_hist.operator == '+')
        
        if (addition_correct and shape_correct and 
            addition_correct_2d and shape_correct_2d and 
            error_raised and history_correct):
            return True
        else:
            return False
    
    def mul_eval(self):
        # Test case 1: Simple 1D tensors
        a = poortorch.tensor([1, 2, 3])
        b = poortorch.tensor([4, 5, 6])
        result = a * b
        
        # Check correct multiplication
        expected = [4, 10, 18]
        multiplication_correct = all(result.__data__[i] == expected[i] for i in range(len(expected)))
        
        # Check shape maintained
        shape_correct = result.shape == (3,)
        
        # Test case 2: 2D tensors
        a_2d = poortorch.tensor([[1, 2], [3, 4]])
        b_2d = poortorch.tensor([[5, 6], [7, 8]])
        result_2d = a_2d * b_2d
        
        # Check correct multiplication
        expected_2d = [[5, 12], [21, 32]]
        multiplication_correct_2d = all(
            result_2d.__data__[i][j] == expected_2d[i][j] 
            for i in range(len(expected_2d)) 
            for j in range(len(expected_2d[0]))
        )
        
        # Check shape maintained
        shape_correct_2d = result_2d.shape == (2, 2)
        
        # Test case 3: Mismatched shapes (should raise Exception)
        a_mismatch = poortorch.tensor([1, 2, 3])
        b_mismatch = poortorch.tensor([4, 5])
        error_raised = False
        try:
            a_mismatch * b_mismatch
        except Exception as e:
            error_raised = "don't have the same shape 😔" in str(e)
        
        # Test case 4: Check history tracking for backward propagation
        a_hist = poortorch.tensor([1.0, 2.0], requires_grad=True)
        b_hist = poortorch.tensor([3.0, 4.0], requires_grad=True)
        result_hist = a_hist * b_hist
        
        history_correct = (result_hist.history == [a_hist, b_hist] and 
                          result_hist.operator == '*')
        
        if (multiplication_correct and shape_correct and 
            multiplication_correct_2d and shape_correct_2d and 
            error_raised and history_correct):
            return True
        else:
            return False
    
    def matmul_eval(self):
        # Test case 1: Matrix multiplication of 2D tensors
        a = poortorch.tensor([[1, 2], [3, 4]])  # 2x2
        b = poortorch.tensor([[5, 6], [7, 8]])  # 2x2
        result = a @ b
        
        # Check correct matrix multiplication
        expected = [[19, 22], [43, 50]]  # (1*5 + 2*7), (1*6 + 2*8), (3*5 + 4*7), (3*6 + 4*8)
        matmul_correct = all(
            result.__data__[i][j] == expected[i][j] 
            for i in range(len(expected)) 
            for j in range(len(expected[0]))
        )
        
        # Check shape is correct for matrix multiplication
        shape_correct = result.shape == (2, 2)
        
        # Test case 2: Matrix multiplication with different dimensions
        a_rect = poortorch.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
        b_rect = poortorch.tensor([[7, 8], [9, 10], [11, 12]])  # 3x2
        result_rect = a_rect @ b_rect
        
        # Check correct matrix multiplication
        expected_rect = [[58, 64], [139, 154]]  # (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12), ...
        matmul_correct_rect = all(
            result_rect.__data__[i][j] == expected_rect[i][j] 
            for i in range(len(expected_rect)) 
            for j in range(len(expected_rect[0]))
        )
        
        # Check shape is correct for matrix multiplication
        shape_correct_rect = result_rect.shape == (2, 2)
        
        # Test case 3: Non-2D tensors (should raise Exception)
        a_1d = poortorch.tensor([1, 2, 3])
        b_1d = poortorch.tensor([4, 5, 6])
        error_raised_1d = False
        try:
            a_1d @ b_1d
        except Exception as e:
            error_raised_1d = "only supported for 2D tensors" in str(e)
        
        # Test case 4: Check history tracking for backward propagation
        a_hist = poortorch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b_hist = poortorch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result_hist = a_hist @ b_hist
        
        history_correct = (result_hist.history == [a_hist, b_hist] and 
                          result_hist.operator == '@')
        
        if (matmul_correct and shape_correct and 
            matmul_correct_rect and shape_correct_rect and 
            error_raised_1d and history_correct):
            return True
        else:
            return False
    
    def float_method_eval(self):
        # Test case 1: Integer tensor converted to float
        int_tensor = poortorch.tensor([1, 2, 3])
        float_result = int_tensor.float()
        
        # Check values are converted to float
        float_conversion_correct = all(
            isinstance(float_result.__data__[i], float) 
            for i in range(len(float_result.__data__))
        )
        
        # Check values are correct
        values_correct = all(
            float_result.__data__[i] == float(int_tensor.__data__[i]) 
            for i in range(len(int_tensor.__data__))
        )
        
        # Check shape maintained
        shape_correct = float_result.shape == int_tensor.shape
        
        # Test case 2: 2D integer tensor converted to float
        int_tensor_2d = poortorch.tensor([[1, 2], [3, 4]])
        float_result_2d = int_tensor_2d.float()
        
        # Check values are converted to float
        float_conversion_correct_2d = all(
            isinstance(float_result_2d.__data__[i][j], float) 
            for i in range(len(float_result_2d.__data__)) 
            for j in range(len(float_result_2d.__data__[0]))
        )
        
        # Check values are correct
        values_correct_2d = all(
            float_result_2d.__data__[i][j] == float(int_tensor_2d.__data__[i][j]) 
            for i in range(len(int_tensor_2d.__data__)) 
            for j in range(len(int_tensor_2d.__data__[0]))
        )
        
        # Check shape maintained
        shape_correct_2d = float_result_2d.shape == int_tensor_2d.shape
        
        # Test case 3: Check history tracking for backward propagation
        tensor_hist = poortorch.tensor([1, 2, 3], requires_grad=True)
        float_result_hist = tensor_hist.float()
        
        history_correct = (float_result_hist.history == [tensor_hist] and 
                          float_result_hist.operator == 'float')
        
        # Test case 4: Already float tensor
        float_tensor = poortorch.tensor([1.5, 2.5, 3.5])
        float_result_already = float_tensor.float()
        
        already_float_correct = all(
            float_result_already.__data__[i] == float_tensor.__data__[i] 
            for i in range(len(float_tensor.__data__))
        )
        
        if (float_conversion_correct and values_correct and shape_correct and
            float_conversion_correct_2d and values_correct_2d and shape_correct_2d and
            history_correct and already_float_correct):
            return True
        else:
            return False
    
    def mean_eval(self):
        # Test case 1: Mean of 1D tensor
        tensor_1d = poortorch.tensor([1, 2, 3, 4, 5], requires_grad=True).float()
        mean_result = poortorch.mean(tensor_1d)
        mean_correct = float(mean_result) == 3.0
        
        # Test case 2: Mean of 2D tensor (all elements)
        tensor_2d = poortorch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True).float()
        mean_result_2d = poortorch.mean(tensor_2d)
        mean_correct_2d = float(mean_result_2d) == 3.5
        
        # Test case 3: Mean along dimension 0
        mean_result_dim0 = poortorch.mean(tensor_2d, dim=0)
        expected_dim0 = [2.5, 3.5, 4.5]  # [mean([1,4]), mean([2,5]), mean([3,6])]
        mean_correct_dim0 = all(
            mean_result_dim0.__data__[i] == expected_dim0[i]
            for i in range(len(expected_dim0))
        )
        
        # Test case 4: Mean along dimension 1
        mean_result_dim1 = poortorch.mean(tensor_2d, dim=1)
        expected_dim1 = [2.0, 5.0]  # [mean([1,2,3]), mean([4,5,6])]
        mean_correct_dim1 = all(
            mean_result_dim1.__data__[i] == expected_dim1[i]
            for i in range(len(expected_dim1))
        )
        
        # Test case 5: Check history tracking for backward propagation
        history_correct = (mean_result.history == [tensor_1d] and 
                          mean_result.operator == 'mean-dim-None')
        
        if self.torch_eval:
            # Compare with torch implementation
            import torch
            torch_tensor_1d = torch.tensor(tensor_1d.__data__, dtype=torch.float32)
            torch_mean = torch.mean(torch_tensor_1d)
            torch_match = abs(torch_mean.item() - float(mean_result)) < 1e-5
            
            torch_tensor_2d = torch.tensor(tensor_2d.__data__, dtype=torch.float32)
            torch_mean_2d = torch.mean(torch_tensor_2d)
            torch_match_2d = abs(torch_mean_2d.item() - float(mean_result_2d)) < 1e-5
            
            torch_mean_dim0 = torch.mean(torch_tensor_2d, dim=0)
            torch_match_dim0 = torch.allclose(
                torch_mean_dim0,
                torch.tensor(mean_result_dim0.__data__, dtype=torch.float32),
                rtol=1e-5
            )
            
            torch_mean_dim1 = torch.mean(torch_tensor_2d, dim=1)
            torch_match_dim1 = torch.allclose(
                torch_mean_dim1,
                torch.tensor(mean_result_dim1.__data__, dtype=torch.float32),
                rtol=1e-5
            )
            
            if (mean_correct and mean_correct_2d and mean_correct_dim0 and 
                mean_correct_dim1 and history_correct and torch_match and 
                torch_match_2d and torch_match_dim0 and torch_match_dim1):
                return True
            else:
                return False
        else:
            if (mean_correct and mean_correct_2d and mean_correct_dim0 and 
                mean_correct_dim1 and history_correct):
                return True
            else:
                return False
    


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "torch": torch_eval = True
    else:
        print("\nA large number of tests are skipped if torch is not used. To run all tests, run the script with 'torch' argument. \n\n")
        torch_eval = False
    if torch_eval:
        import torch
    evaluator = EvaluatePoortorch(torch_eval=torch_eval)
    evaluator.eval()
    print("Evaluation complete.")