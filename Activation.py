import numpy as np
class Activation:
    def __init__(self):
        pass

  
class Relu_activation(Activation):
   

    def activate_forward(self,inputs):
        self.output=np.maximum(0,inputs)

    def back_propagation(self,dinputs):
        self.doutputs[dinputs<=0]=0
        return self.doutputs
        

        



class Softmax_activation(Activation):

    def activate_forward(self,inputs):
        scale_down=inputs-np.max(inputs,axis=1,keepdims=True)
        exp_inputs=np.exp(scale_down)
        self.output=exp_inputs/np.sum(exp_inputs,axis=1,keepdims=True)
    


    def back_propagation(self, dinputs):
        self.dinputs = np.empty_like(dinputs)
        for index, (single_output, single_dinput) in enumerate(zip(self.output, dinputs)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dinput)

        return self.dinputs
