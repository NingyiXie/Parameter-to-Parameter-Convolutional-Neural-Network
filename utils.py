import numpy as np
import torch

def normalize(parameters):
    p = len(parameters)//2
    input = np.zeros((1,1,2,p))
    input[0][0][0] = np.array(parameters[:p])/np.pi
    input[0][0][1] = np.array(parameters[p:])/(np.pi/2)
    return input

def recover(prediction):
    params = prediction.squeeze(0).squeeze(0).clone().cpu().numpy()
    gammas = params[0]*np.pi
    betas = params[1]*np.pi/2
    return gammas.tolist()+betas.tolist()

def predict(input,ppn,t):
    input = torch.tensor(input,dtype=torch.float).to(torch.device("cpu"))
    ppn.eval()
    return ppn.predict(input,t)