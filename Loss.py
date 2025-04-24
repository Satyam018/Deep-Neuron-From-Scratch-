import numpy as np


class Loss:
    def __init__(self):
        pass



class Cross_entropy_loss(Loss):


#      y_pred  can be y_true=[rightind1,rightind2,tightind3] or y_true=[[1,0,0,],[0,1,0],[0,0,1]]
    def calculate_loss(self,y_true,y_predicted):

        y_true=np.array(y_true)
        y_predicted_clip=np.clip(y_predicted,1e-19,1-1e-19)
        if len(y_true.shape)==1:
            total_sample=len(y_true)
            output_prediction=[y_predicted_clip[i][y_true[i]] for i in range(total_sample)]
        else:
            output_prediction=np.sum(y_predicted_clip*y_true,axis=1)
        
        loss=-np.log(output_prediction)
        return np.mean(loss)




if __name__=="__main__":
    y_true=[0,1,2]
    y_predicted=[[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]
    loss=Cross_entropy_loss()
    print(loss.calculate_loss(y_true,y_predicted))

   