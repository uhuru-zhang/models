import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np

HIDDEN_SIZE = 100

class Net(nn.Module):

    # def __init__(self):
    #     super(Net, self).__init__()
    #     # 1 input image channel, 6 output channels, 5x5 square convolution
    #     # kernel
    #     #self.bn0 = nn.BatchNorm1d(81)
    #     self.fc0 = nn.Linear(9*9,28*28)
    #     #self.bn1 = nn.BatchNorm1d(28*28)
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     #self.bn2 = nn.BatchNorm2d(6)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     #self.bn3 = nn.BatchNorm2d(16)
    #     # an affine operation: y = Wx + b
    #     self.fc1 = nn.Linear(16 * 4 * 4, 100)
    #     #self.bn4 = nn.BatchNorm1d(100)
    #     self.fc2 = nn.Linear(100, 81)
    #     #self.bn5 = nn.BatchNorm1d(81)
    #     self.fc3 = nn.Linear(81, 10)
    #
    # def forward(self, x):
    #     x = x.view(-1, self.num_flat_features(x))
    #    # x = self.bn0(x)
    #     x = F.relu(self.fc0(x))
    #    # x = self.bn1(x)
    #     x = x.view(-1 ,1, 28,28)
    #     # Max pooling over a (2, 2) window
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #     # If the size is a square you can only specify a single number
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.relu(self.fc1(x))
    #     #x = self.bn4(x)
    #     x = F.relu(self.fc2(x))
    #     #x = self.bn5(x)
    #     x = F.softmax(self.fc3(x))
    #     return x

    def __init__(self):
       super(Net, self).__init__()
       mlp_fcs = [nn.Linear(in_features=81, out_features=HIDDEN_SIZE)]
       mlp_bns = [nn.BatchNorm1d(num_features=81)]

       for i in range(1, 3):
           mlp_fcs.append(nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
           mlp_bns.append(nn.BatchNorm1d(num_features=HIDDEN_SIZE))

       self.mlp_fc_end = nn.Linear(in_features=HIDDEN_SIZE, out_features=10)

       self.mlp_fcs = nn.ModuleList(mlp_fcs)
       self.mlp_bns = nn.ModuleList(mlp_bns)

    def forward(self, x):
       x = x.view(-1, self.num_flat_features(x))
       for i in range(len(self.mlp_fcs)):
           x = self.mlp_bns[i](x)
           x = self.mlp_fcs[i](x)
           x = F.relu(x)

       x = self.mlp_fc_end(x)

       return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#net = Net()
#print(net)

#data
trainfile = open('./2/TrainSamples.csv')
#train = pd.read_csv(trainfile, skiprows=14000, nrows=None, header=None).iloc[0:,0:].as_matrix().reshape(-1,81)
#print(train[0])

trainLabelfile = open('./2/TrainLabels.csv')
#trainLabel = pd.read_csv(trainLabelfile,header=None).iloc[0:,0:].as_matrix().reshape(-1,1)
testfile = open('./2/TrainSamples.csv')
testLabelfile = open('./2/TrainLabels.csv')

class trainDataset(Dataset):
    def __init__(self, csv_file, csv_file2, transform=None, train=True, normalization=True):
        skiprows, nrows = (0, 14000) if train else (14000, None)
        self.train = pd.read_csv(csv_file, skiprows=skiprows, dtype=np.float, nrows=nrows,header = None).values
        self.label = pd.read_csv(csv_file2, skiprows=skiprows, dtype=np.float, nrows=nrows,header = None).values
        self.transform = transform
        
        if normalization:
            self.train = (self.train - np.mean(self.train)) / np.std(self.train, ddof=1)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        data = self.train[idx].reshape(9, 9)
        target = self.label[idx][0]
        sample = {'data': data,'target' : target}

        if self.transform:
            sample = self.transform(sample)

        return sample

#test = trainDataset(csv_file=trainfile,csv_file2=trainLabelfile)

class ToTensor(object):
    def __call__(self, sample):
        data,target = sample['data'],sample['target']

        return {'data':torch.from_numpy(data).float(),
                'target':target}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# )
model = Net().to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

def main():
    traintransformed_data = trainDataset(csv_file=trainfile,csv_file2=trainLabelfile,transform=ToTensor(),train = True)

    traindataloader = DataLoader(traintransformed_data, batch_size=256,
                            shuffle=True, num_workers=32)

#testtransformed_data = trainDataset(csv_file=trainfile,csv_file2=trainLabelfile,transform=transforms.Compose([
#                           ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                       ]),train = False)

#testdataloader = DataLoader(traintransformed_data, batch_size=4,
#                        shuffle=True, num_workers=0)

    testtransformed_data = trainDataset(csv_file=testfile,csv_file2=testLabelfile,transform=ToTensor(),train = False)

    testdataloader = DataLoader(testtransformed_data, batch_size=256,
                            shuffle=True, num_workers=32)

    # create your optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=10)

    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, 100):
        if epoch == 20:
            optimizer = optim.Adadelta(model.parameters(), lr=1)
        if epoch == 30:
            optimizer = optim.Adadelta(model.parameters(), lr=0.1)
        if epoch == 40:
            optimizer = optim.Adadelta(model.parameters(), lr=0.01)
        for step, sample in enumerate(traindataloader):
            input = sample['data'].to(device).float()
            target = sample['target'].to(device).long()
            #print(input.size(),target.size())
            optimizer.zero_grad()   # zero the gradient buffers
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update

            #test
            if(step%10 == 0):
                print("Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {}".format(
                    epoch, step * len(sample), len(traindataloader.dataset),
                    100. * step / len(traindataloader), loss.item()
                    ))

                with torch.no_grad():
                    model.eval()
                    test_loss = 0
                    correct = 0

                    for sample in testdataloader:
                        input = sample['data'].to(device).float()
                        target = sample['target'].to(device).long()
                        output = model(input)

                        test_loss += criterion(output, target)
                        pred = output.max(1, keepdim=True)[1]
                        # target = target.max(1, keepdim=True)[1]
                        correct += pred.eq(target.view_as(pred)).sum().item()

                    test_loss /= len(testdataloader.dataset)

                    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
                            .format(test_loss, correct, len(testdataloader.dataset),
                                    100. * correct / len(testdataloader.dataset)))

if __name__ == '__main__':
    main()


#问题1
#同时读取一个文件出错
#问题2
# datasets.MNIST(root="~/data_set", train=False, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                       ]))
#问题3
#gpu运行
#问题4
#test_loss += F.nll_loss(input=output, target=target)