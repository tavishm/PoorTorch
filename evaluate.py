# -*- coding: utf-8 -*-
from main import poortorch
import math
import sys



class evaluate_poortorch():
    def __init__(self, torch_eval=True):
        self.functions_to_evaluate = {

            # Properly checkable functions
            poortorch.rand: self.rand_eval,
            poortorch.randn: self.randn_eval,
            poortorch.zeros: self.zeros_eval,
            poortorch.exp: self.exp_eval,
            poortorch.log: self.log_eval,
            poortorch.tanh: self.tanh_eval,

        }
        if torch_eval:
            self.functions_to_evaluate.update({
                poortorch.mean: self.mean_eval,
            })

    def eval(self):
        for poortorch_f, eval_f in self.functions_to_evaluate.items():
            passed = eval_f()
            print(f"{poortorch_f.__name__} evaluation result: {"✅" if passed else "❌"}")

    def _check_condition(xl: list, condition_f):
        if isinstance(xl, (int, float)):
            return condition_f(xl)
        else:
            truth_list = []
            for xi in xl:
                truth_list.append(evaluate_poortorch._check_condition(xi, condition_f))
            return all(truth_list)

    def _check_operation_condition(xl_o: list, xl_f: list, operation_f):
        if isinstance(xl_o, (int, float)) and isinstance(xl_f, (int, float)):
            return operation_f(xl_o, xl_f)
        else:
            truth_list = []
            for xi_o, xi_f in zip(xl_o, xl_f):
                truth_list.append(evaluate_poortorch._check_operation_condition(xi_o, xi_f, operation_f))
            return all(truth_list)

    def rand_eval(self):
        test_tensor = poortorch.rand((1,2,3,4), v_range=(7,8), requires_grad=True)
        in_range = evaluate_poortorch._check_condition(test_tensor.__data__, lambda k: 7<=k<=8)
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
        in_range = evaluate_poortorch._check_condition(test_tensor.__data__, lambda k: k==0)
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==True and in_range:
            return True
        else:
            return False
        
    def exp_eval(self):
        xt_o = poortorch.randn((1,2,3,4), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.exp(xt_o)
        in_range = evaluate_poortorch._check_operation_condition(xt_o.__data__, test_tensor.__data__, lambda x_o, x_f: x_f == math.exp(x_o))
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==False and in_range:
            return True
        else:
            return False
        
    def log_eval(self):
        xt_o = poortorch.randn((1,2,3,4), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.log(xt_o)
        in_range = evaluate_poortorch._check_operation_condition(xt_o.__data__, test_tensor.__data__, lambda x_o, x_f: x_f == math.log(x_o) if x_o > 0 else math.isnan(x_f)) 
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==False and in_range:
            return True
        else:
            return False
    
    def tanh_eval(self):
        xt_o = poortorch.randn((1,2,3,4), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.tanh(xt_o)
        in_range = evaluate_poortorch._check_operation_condition(xt_o.__data__, test_tensor.__data__, lambda x_o, x_f: x_f == math.tanh(x_o))
        if test_tensor.shape == (1,2,3,4) and test_tensor.requires_grad==False and in_range:
            return True
        else:
            return False
        
    def mean_eval(self):
        # Along none-dim
        xt_o = poortorch.randn((1,2,3,4,5), mean=1, std=2, requires_grad=True)
        test_tensor = poortorch.mean(xt_o)
        xp_o = torch.tensor(xt_o.__data__)
        torch_mean = torch.mean(xp_o)

        if test_tensor.requires_grad == False and abs(float(torch_mean) - float(test_tensor)) < 0.001:
            # Along dim 1
            test_tensor_dim1 = poortorch.mean(xt_o, dim=1)
            torch_mean_dim1 = torch.mean(xp_o, dim=1)
            if test_tensor_dim1.requires_grad == False and torch.allclose(torch_mean_dim1, torch.tensor(test_tensor_dim1.__data__)):
                # Along dim 2
                test_tensor_dim2 = poortorch.mean(xt_o, dim=2)
                torch_mean_dim2 = torch.mean(xp_o, dim=2)
                if test_tensor_dim2.requires_grad == False and torch.allclose(torch_mean_dim2, torch.tensor(test_tensor_dim2.__data__)):
                    # Along dim 3
                    test_tensor_dim3 = poortorch.mean(xt_o, dim=3)
                    torch_mean_dim3 = torch.mean(xp_o, dim=3)
                    if test_tensor_dim3.requires_grad == False and torch.allclose(torch_mean_dim3, torch.tensor(test_tensor_dim3.__data__)):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
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
    evaluator = evaluate_poortorch(torch_eval=torch_eval)
    evaluator.eval()
    print("Evaluation complete.")