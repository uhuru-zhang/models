import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from models.BN_Network import ExpDataSet

HIDDEN_SIZE = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        mlp_fcs = [nn.Linear(in_features=81, out_features=HIDDEN_SIZE)]
        mlp_bns = [nn.BatchNorm1d(num_features=100)]

        for i in range(1, 3):
            mlp_fcs.append(nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE))
            mlp_bns.append(nn.BatchNorm1d(num_features=HIDDEN_SIZE))

        self.mlp_fc_end = nn.Linear(in_features=HIDDEN_SIZE, out_features=10)

        self.mlp_fcs = nn.ModuleList(mlp_fcs)
        self.mlp_bns = nn.ModuleList(mlp_bns)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for i in range(len(self.mlp_fcs)):
            x = self.mlp_fcs[i](x)
            x = self.mlp_bns[i](x)
            x = F.relu(x)

        x = self.mlp_fc_end(x)

        return F.log_softmax(x, dim=1)


def main():
    train_loader = D.DataLoader(
        ExpDataSet(),
        batch_size=256, shuffle=True, num_workers=32
    )

    test_loader = D.DataLoader(
        ExpDataSet(train=False),
        batch_size=64, shuffle=True, num_workers=32
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adadelta(params=model.parameters(), lr=10)

    for epoch in range(1, 100):

        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).long()

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(input=output, target=target)
            loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {}".format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                           100. * batch_index / len(train_loader), loss.item()
                ))

        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0

            for data, target in test_loader:
                data, target = data.to(device).float(), target.to(device).long()
                output = model(data)

                test_loss += F.nll_loss(input=output, target=target)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))

    # Visualize the STN transformation on some input batch


if __name__ == '__main__':
    main()
