import numpy as np

def recover_parameters(predict_angle):
    params = predict_angle.squeeze(0).squeeze(0).clone().cpu().numpy()
    gammas = params[0]*np.pi
    betas = params[1]*np.pi/2
    return gammas.tolist()+betas.tolist()