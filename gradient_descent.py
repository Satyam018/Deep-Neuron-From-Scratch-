class GDOptimiser:
    def __init__(self,learning_rate=1,decay=0.01):
        self.learning_rate=learning_rate
        self.decay=decay
        self.current_learning_rate=learning_rate
    

    

    def update(self,layer):
        if self.decay:
            self.current_learning_rate=self.learning_rate/(1+self.decay*layer.iteration)
        layer.weights+=-self.current_learning_rate*layer.dweights
        layer.biases+=-self.current_learning_rate*layer.dbiases
    



    





class GDOptimiser

