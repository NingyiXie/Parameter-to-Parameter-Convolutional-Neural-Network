import torch.nn as nn

class RB(nn.Module):
    def __init__(self,in_channels):
        super(RB,self).__init__()
        # two 3x3 convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x=self.act(self.conv2(self.act(self.conv1(x)))+x)
        return x

class PPN(nn.Module):
    def __init__(self, D=4, t=4):
        super(PPN, self).__init__()
        # up-sampling
        self.us1 = nn.Conv2d(1, 16, kernel_size=2, padding=1)
        self.us2 = nn.Conv2d(16, 64, kernel_size=2, padding=1)
        self.act = nn.ReLU(inplace = True)
        
        # residual blocks
        self.D=D
        RBs=[]
        for _ in range(self.D):
            RBs.append(RB(64))
        
        self.RBs_list=nn.ModuleList(RBs)
        
        # down-sampling
        self.ds = nn.Conv2d(64, 1, (3,2), padding=0)
        
        # recursive t times when training
        self.t = t
                    
    def forward(self, x):
        # Forward propagation of training
        out = []
        for _ in range(self.t): # t ouputs
            x = self.act(self.us2(self.act(self.us1(x))))
            for i in range(self.D):
                x = self.RBs_list[i](x)
            x = self.ds(x)
            out.append(x)
        return out
    
    def predict(self, x, t):
        # input data and designate t times of recursion for predicting desired depth QAOA. 
        for _ in range(t):
            x = self.act(self.us2(self.act(self.us1(x))))
            for i in range(self.D):
                x = self.RBs_list[i](x)
            x = self.ds(x)
        return x