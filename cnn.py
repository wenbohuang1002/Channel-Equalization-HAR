import torch
import torch.nn as nn
import torch.nn.functional as F
from devkit.ops import CE
from atten import ChannelAttention

" UCI-HAR "
class UCI(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.fc = nn.Linear(256 * 2 * 9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(1))

        return out

class UCI_ATT(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.ca3 = ChannelAttention(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.fc = nn.Linear(256 * 2 * 9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ca1(out)*out
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca2(out) * out
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.ca3(out) * out
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        # out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE_L1(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE_L1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE_L2(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE_L2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        # out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE_N1(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE_N1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE_N2(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE_N2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE_N3(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE_N3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        # self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        # out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class UCI_CE_Na(nn.Module):
    def __init__(self,  num_class=6):
        super(UCI_CE_Na, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256*2*9, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

def test_UCI():
    Baseline = UCI(num_class=6)
    CE = UCI_CE(num_class=6)
    x = torch.randn(64, 1, 128, 9)
    y = Baseline(x)
    z = CE(x)
    print(y.size())
    print(z.size())
" UCI-HAR "

" OPPORTUNITY "
class OPPORTUNITY(nn.Module):
    def __init__(self,  num_class=18):
        super(OPPORTUNITY, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_ATT(nn.Module):
    def __init__(self,  num_class=18):
        super(OPPORTUNITY_ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.ca3 = ChannelAttention(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ca1(out)*out
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca2(out) * out
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.ca3(out) * out
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE_L1(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE_L1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE_L2(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE_L2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        # out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(1)
        # out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE_N1(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE_N1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(1)
        out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE_N2(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE_N2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(1)
        out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE_N3(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE_N3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        # out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(1)
        out = F.normalize(out.cuda(0))

        return out

class OPPORTUNITY_CE_Na(nn.Module):
    def __init__(self, num_class=18):
        super(OPPORTUNITY_CE_Na, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce1 = CE(num_features=64, pooling=False, num_channels=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce2 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 1 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.ce2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(1)
        out = F.normalize(out.cuda(0))

        return out

def test_OPPORTUNITY():
    Baseline = OPPORTUNITY(num_class=18)
    CE = OPPORTUNITY_CE(num_class=18)
    x = torch.randn(64, 1, 40, 113)
    y = Baseline(x)
    z = CE(x)
    print(y.size())
    print(z.size())
" OPPORTUNITY "

" PAMAP2 "
class PAMAP2(nn.Module):
    def __init__(self,  num_class=12):
        super(PAMAP2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.fc = nn.Linear(512 * 3 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class PAMAP2_ATT(nn.Module):
    def __init__(self,  num_class=12):
        super(PAMAP2_ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(True)
        self.ca1 = ChannelAttention(128)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.ca2 = ChannelAttention(256)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.ca3 = ChannelAttention(512)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.fc = nn.Linear(512 * 3 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ca1(out)*out
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca2(out) * out
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.ca3(out) * out
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class PAMAP2_CE(nn.Module):
    def __init__(self, num_class=12):
        super(PAMAP2_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 3 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class PAMAP2_CE_R(nn.Module):
    def __init__(self, num_class=12):
        super(PAMAP2_CE_R, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        # self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.ce1 = CE(num_features=128, pooling=False, num_channels=128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce2 = CE(num_features=256, pooling=False, num_channels=256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 3 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu1(out)
        out = self.pool1(out)
        out = self.ce1(out)
        h1 = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        # out = self.ce2(out)
        h2 = out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)
        h3 = out

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out,h1,h2,h3

class PAMAP2_R(nn.Module):
    def __init__(self, num_class=12):
        super(PAMAP2_R, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        # self.ce3 = CE(num_features=512, pooling=False, num_channels=512)

        self.fc = nn.Linear(512 * 3 * 4, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        h1 = out

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        h2 = out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        # out = self.ce3(out)
        h3 = out

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out,h1,h2,h3

def test_PAMAP2():
    Baseline = PAMAP2(num_class=12)
    CE = PAMAP2_CE(num_class=12)
    R = PAMAP2_CE_R(num_class=12)
    x = torch.randn(64, 1, 86, 120)
    y = Baseline(x)
    z = CE(x)
    r1, h1, h2, h3 = R(x)
    r2, c1, c2, c3 = R(x)
    # h1 = h1.detach().numpy()
    print(y.size())
    print(z.size())
    print(r1.size(), h1.size(), h2.size(), h3.size())
    print(r2.size(), c1.size(), c2.size(), c3.size())
" PAMAP2 "

" WISDM "
class WISDM(nn.Module):
    def __init__(self,  num_class=6):
        super(WISDM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc = nn.Linear(256 * 7 * 3, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class WISDM_ATT(nn.Module):
    def __init__(self,  num_class=6):
        super(WISDM_ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.ca3 = ChannelAttention(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc = nn.Linear(256 * 7 * 3, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ca1(out)*out
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca2(out) * out
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.ca3(out) * out
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class WISDM_CE(nn.Module):
    def __init__(self, num_class=6):
        super(WISDM_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256 * 7 * 3, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

def test_WISDM():
    Baseline = WISDM(num_class=6)
    CE = WISDM_CE(num_class=6)
    x = torch.randn(64, 1, 200, 3)
    y = Baseline(x)
    z = CE(x)
    print(y.size())
    print(z.size())
" WISDM "

" USC-HAD "
class USC(nn.Module):
    def __init__(self,  num_class=12):
        super(USC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.fc = nn.Linear(256 * 8 * 6, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class USC_ATT(nn.Module):
    def __init__(self,  num_class=12):
        super(USC_ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.ca3 = ChannelAttention(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.fc = nn.Linear(256 * 8 * 6, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ca1(out)*out
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca2(out) * out
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.ca3(out) * out
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class USC_CE(nn.Module):
    def __init__(self, num_class=12):
        super(USC_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256 * 8 * 6, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

class USC_C(nn.Module):
    def __init__(self,  num_class=12):
        super(USC_C, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.fc = nn.Linear(256 * 8 * 6, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        h = out

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out, h

class USC_CE_C(nn.Module):
    def __init__(self, num_class=12):
        super(USC_CE_C, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256 * 8 * 6, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)
        h = out

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out, h

def test_USC():
    Baseline = USC(num_class=12)
    CE = USC_CE(num_class=12)
    Baseline_C = USC_C(num_class=12)
    CE_C = USC_CE_C(num_class=12)
    x = torch.randn(64, 1, 512, 6)
    y = Baseline(x)
    z = CE(x)
    y_c, h1 = Baseline_C(x)
    z_c, h2 = CE_C(x)
    print(y.size())
    print(z.size())
    print(y_c.size(), h1.size())
    print(z_c.size(), h2.size())
" USC-HAD "

" UNIMIB "
class UNIMIB(nn.Module):
    def __init__(self,  num_class=17):
        super(UNIMIB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc = nn.Linear(256 * 5 * 3, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class UNIMIB_ATT(nn.Module):
    def __init__(self,  num_class=17):
        super(UNIMIB_ATT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.ca1 = ChannelAttention(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.ca2 = ChannelAttention(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        self.ca3 = ChannelAttention(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.fc = nn.Linear(256 * 5 * 3, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.ca1(out)*out
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.ca2(out) * out
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.ca3(out) * out
        out = self.pool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = nn.LayerNorm(out.size())(out.cpu())
        # out = out.cuda(0)
        out = F.normalize(out.cuda(0))

        return out

class UNIMIB_CE(nn.Module):
    def __init__(self, num_class=17):
        super(UNIMIB_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(256 * 5 * 3, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

def test_UNIMIB():
    Baseline = UNIMIB(num_class=17)
    CE = UNIMIB_CE(num_class=17)
    x = torch.randn(64, 1, 151, 3)
    y = Baseline(x)
    z = CE(x)
    print(y.size())
    print(z.size())
" UNIMIB "

# test_UCI()  # Net_Train.py --dataset uci --lr 5e-4 --batch_size 128 --epochs 200 #
#
# test_OPPORTUNITY()  # Net_Train.py --dataset opportunity --lr 5e-4 --batch_size 64 --epochs 200 #
#
# test_PAMAP2()  # Net_Train.py --dataset pamap2 --lr 5e-4 --batch_size 64 --epochs 200 #
#
# test_WISDM()  # Net_Train.py --dataset wisdm --lr 5e-4 --batch_size 128 --epochs 200 #
#
# test_USC()  # Net_Train.py --dataset usc --lr 5e-4 --batch_size 128 --epochs 200 #
#
# test_UNIMIB()  # Net_Train.py --dataset unimib --lr 5e-4 --batch_size 64 --epochs 200 #

