class GDOptimiser:
    def __init__(self,learning_rate=1,decay=0.01,momentum=0.6):
        self.learning_rate=learning_rate
        self.decay=decay
        self.current_learning_rate=learning_rate
        self.momentum=momentum

    

    

    def update(self,layer):
        if self.decay:
            self.current_learning_rate=self.learning_rate/(1+self.decay*layer.iteration)
        if self.momentum:
            if not hasattr(layer,'momentum_weight'):
                layer.momentum_weight=np.zeros_like(layer.weights)
                layer.momentum_bias=np.zeros_like(layer.biases)
            
            weight_update=self.momentum*layer.momentum_weight-self.current_learning_rate*layer.dweights
            bias_update=self.momentum*layer.momentum_bias-self.current_learning_rate*layer.dbiases
            layer.momentum_weight=weight_update
            layer.momentum_bias=bias_update

            layer.weights+=weight_update
            layer.biases+=bias_update
            
        else:
            layer.weights-=self.learning_rate*layer.dweights
            layer.biases-=self.learning_rate*layer.dbiases

        
    
    


    





class GDOptimiser

