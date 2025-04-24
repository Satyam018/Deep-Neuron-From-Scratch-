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
    # Assumes dinputs is the true labels (one-hot or adjusted gradient)
        self.doutputs = self.output - dinputs
        return self.doutputs
