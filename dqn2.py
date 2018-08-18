from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T
from torch import nn
from torch.nn import functional as F

class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action
        # (-1, 4, 84, 84) -> (-1, 32, 20, 20) 
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        # (-1, 32, 20, 20) -> (-1, 64, 9, 9)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        # (-1, 64, 9, 9) -> (-1, 64, 7, 7)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        # (-1, 7*7*64) -> (-1, 512)
        self.linear1 = nn.Linear(7*7*64, 512)
        # (-1, 512) -> (-1, n_action)
        self.linear2 = nn.Linear(512, self.n_action)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = h.view(h.size(0), -1) # flatten the features
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h

