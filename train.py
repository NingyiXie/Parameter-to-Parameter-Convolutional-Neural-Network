import numpy as np
import json,os,argparse

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

from models import PPN

class GraphDataset(data.Dataset):
    def __init__(self, seeds_dict):
        super(GraphDataset, self).__init__()
        # graph info list
        self.graphs = []
        for n in seeds_dict:
            for t in seeds_dict[n]:
                for seed in seeds_dict[n][t]:
                    self.graphs.append(f'n{n}_{t}_seed{seed}')  # a string contains node number, edge probability, graph seed
    def __getitem__(self, index):
        return self.graphs[index]
    def __len__(self):
        return len(self.graphs)        

def getBatchData(graph_list,folder,ps,pt,device):
    angles = []
    for p in range(ps,pt+1):

        # intial a parameter array with zero
        angle = np.zeros((len(graph_list),1,2,p))

        for idx,graph in enumerate(graph_list):

            # read data from json
            path = f'{folder}{graph}.json'
            with open(path, "r") as json_file:
                data = json.load(json_file)
            params = data['params'][str(p)]

            # normalization
            angle[idx][0][0] = np.array(params[:p])/np.pi
            angle[idx][0][1] = np.array(params[p:])/(np.pi/2)
        angles.append(torch.tensor(angle,dtype=torch.float).to(device))
    
    return angles # return one batch of angels


def main():
    parser = argparse.ArgumentParser(description='Train')

    # load training info from a json file
    parser.add_argument('-opt', type=str, help='Path to options JSON file.',default='./opt/opt1.json')
    f=open(parser.parse_args().opt,encoding='utf-8')
    content = f.read()
    opt = json.loads(content)

    # designate device: cpu or cuda
    device = torch.device(opt['device'])

    # epochs, batchsize,learning rate
    number_of_epochs=opt["epoch"]
    batch_size=opt["batch_size"]
    lr=opt['learning_rate']

    
    graph_seeds_path = opt['graph_seeds_path']
    with open(graph_seeds_path, "r") as json_file:
        seeds_dict = json.load(json_file)
    # the folder save the optimal parameters of graphs for training
    data_folder = opt['data_folder']

    # the training parameters are from depth ps to pt
    ps = opt["ps"]
    pt = opt["pt"]

    # D residul blocks
    D = opt['D']

    # model file name
    name = opt['name']
    # model path
    tmp_model_path = opt["tmp_folder"]+name+'.pth'
    save_model_path = opt["save_folder"]+name+'.pth'

    # mkdir
    if not os.path.exists(opt["tmp_folder"]):
        os.makedirs(opt["tmp_folder"])
    if not os.path.exists(opt["save_folder"]):
        os.makedirs(opt["save_folder"])

    # path of tensorboard log
    # log_dir = opt["log"]

    seed = opt.get("seed",np.random.randint(1,10000))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # loss fucntion : mse
    lossF = nn.MSELoss(reduction='mean')

    # use cuda or cput
    device = torch.device(device)

    #load PPN
    net = PPN(D=D,t=pt-ps)
    net=net.to(device)

    i=0 # count update times
    start_epoch=1 # epoch num of start

    # adam optimizer
    if lr!=0:
        optimizer = optim.Adam(net.parameters(),lr=lr)

    # if tmp model exists, load the model
    if os.path.exists(tmp_model_path):
        checkpoint = torch.load(tmp_model_path)
        net.load_state_dict(checkpoint['net'])

        # if lr is not set, load the optimizer
        if lr==0:
            optimizer = optim.Adam(net.parameters(),lr=1e-4)
            optimizer.load_state_dict(checkpoint['optimizer'])

        # else adopt new learning rate
        else: 
            optimizer = optim.Adam(net.parameters(),lr=lr)

        # load training info
        i=checkpoint['i']
        start_epoch=checkpoint['epoch']+1
        
    #tensorboard
    # writer=SummaryWriter(log_dir)

    # train loader
    graph_set = GraphDataset(seeds_dict['train'])
    trainloader = DataLoader(dataset=graph_set, batch_size=batch_size, shuffle=True)
    
    # training ....
    for epoch in range(start_epoch,number_of_epochs+start_epoch):

        # cache = {'loss': 0}

        for batch in trainloader:
            # each batch is a graph list    
            angles = getBatchData(batch,data_folder,ps,pt,device)  # angles contains parameters from ps to pt depth

            predict_angles = net(angles[0]) # predict_angles is a list contains the predictions from ps+1 to pt depth
            
            loss = 0
            for angle_idx in range(1,len(angles)):
                loss += lossF(predict_angles[angle_idx-1], angles[angle_idx])  # sum all loss of pt-ps objectives
            loss = loss/(len(angles)-1)  # averaged loss

            # update
            net.zero_grad()
            loss.backward()
            optimizer.step()

            # cache['loss'] += loss.item()

            # log when parameter set of PPN update
            # writer.add_scalars('scalar/trainset_loss',{f'{name}_TrainSet_Loss':loss.item()},i)

            i+=1
        
        # train_avg_loss = cache['loss']/(len(trainloader))

        # print epoch
        print(i, " Epoch: ", epoch)

        # log each epoch's averaged loss
        # writer.add_scalars('scalar/avg_loss',{f'{name}_TrainSet_Avg_Loss':train_avg_loss},epoch)
        

        #save tmp model
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(),  'i': i, 'epoch':epoch}
        torch.save(state, tmp_model_path)

    # save the model when training finish
    torch.save(net.state_dict(), save_model_path)


if __name__ == '__main__':
    main()