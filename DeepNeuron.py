import numpy as np

inputs=[[1.2,3.4,5.6],
        [7.8,9.0,1.2]]

class Dense_layer:
    def __init__(self,input_size,neuron_number):
        weights=0.1*np.random.rand(neuron_number,input_size)
        self.weights=weights.T
        self.biases=np.zeros(neuron_number)



    def forward(self,inputs):
        weighted_input=np.dot(inputs,self.weights)
        self.output=weighted_input+self.biases

        

class Relu_activation:
    def __init__(self):
        pass

    def activate_forward(self,inputs):
        self.output=np.maximum(0,inputs)
        

   





layer1=Dense_layer(3,3)
layer1.forward(inputs)

relu=Relu_activation()
relu.activate_forward(layer1.output)
print(relu.output)




