import numpy as np
from Activation import Relu_activation,Softmax_activation
from Loss import Cross_entropy_loss

inputs=[[1.2,3.4,5.6],
        [7.8,9.0,1.2]]

y_true=[0,1]

class Dense_layer:
    def __init__(self,input_size,neuron_number):
        weights=0.1*np.random.rand(neuron_number,input_size)
        self.weights=weights.T
        self.biases=np.zeros(neuron_number)
    



    def forward(self,inputs):
        weighted_input=np.dot(inputs,self.weights)
        self.output=weighted_input+self.biases

        


        

        

   





layer1=Dense_layer(3,3)
layer1.forward(inputs)


relu=Relu_activation()
relu.activate_forward(layer1.output)

layer2=Dense_layer(3,2)
layer2.forward(relu.output)
softmax=Softmax_activation()
softmax.activate_forward(layer2.output)

loss=Cross_entropy_loss()
print('loss',loss.calculate_loss(y_true,softmax.output))



# print('softmax',softmax.output)







