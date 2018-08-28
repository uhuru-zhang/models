import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as D
import torch.optim as optim

"""
highway network
https://arxiv.org/pdf/1505.00387.pdf

"""


class HighWayMLP(nn.Module):
    def __init__(self, input_size, out_put_size, gate_bias=-2, activation_function=F.relu, gate_activation=F.softmax):
        super(HighWayMLP, self).__init__()

        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.highway_layers = nn.ModuleList([
            nn.ModuleDict({
                "H": nn.Linear(in_features=input_size, out_features=input_size),
                "T": nn.Linear(in_features=input_size, out_features=input_size),

            })
            for _ in range(200)
        ])

        for layer in self.highway_layers:
            layer["T"].bias.data.fill_(gate_bias)

        self.plain_out = nn.Linear(in_features=input_size, out_features=out_put_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        for layer in self.highway_layers:
            H, T = self.activation_function(layer["H"](x)), self.gate_activation(layer["T"](x))

            x = torch.mul(H, T) + torch.mul(x, 1 - T)

        x = self.plain_out(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    train_loader = D.DataLoader(
        datasets.MNIST(root="~/data_set", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=256, shuffle=True, num_workers=32
    )

    test_loader = D.DataLoader(
        datasets.MNIST(root="~/data_set", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, num_workers=32
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HighWayMLP(input_size=28 * 28, out_put_size=10).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adadelta(params=model.parameters())

    for epoch in range(1, 100):

        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

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
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += F.nll_loss(input=output, target=target)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))
