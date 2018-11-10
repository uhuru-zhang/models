import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.mlp_1_f = nn.Linear(in_features=320, out_features=50)
        self.mlp_2_f = nn.Linear(in_features=50, out_features=10)

        # input =  # n * 1 * 28 * 28
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7),  # n * 8 * 22 * 22
            nn.MaxPool2d(kernel_size=2, stride=2),  # n * 8 * 11 * 11
            nn.ReLU(inplace=True),  # n * 8 * 11  * 11
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),  # n * 10 * 7 * 7
            nn.MaxPool2d(kernel_size=2, stride=2),  # n * 10 * 3 * 3
            nn.ReLU(inplace=True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=10 * 3 * 3, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta=theta, size=x.size())
        x = F.grid_sample(input=x, grid=grid)

        return x

    def forward(self, x):
        x = self.stn(x=x)

        x = F.relu(F.max_pool2d(input=self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(input=self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.mlp_1_f(x))
        x = F.dropout(x, training=self.training)
        x = self.mlp_2_f(x)

        return F.log_softmax(x, dim=1)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = D.DataLoader(
        datasets.MNIST(root="~/data_set", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=5, shuffle=True, num_workers=32
    )

    test_loader = D.DataLoader(
        datasets.MNIST(root="~/data_set", train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, num_workers=32
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(params=model.parameters())

    def train(epoch):
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(input=output, target=target)
            loss.backward()
            optimizer.step()

            if batch_index % 500 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f})]\tLoss".format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                           100. * batch_index / len(train_loader), loss.item()
                ))

    def test():
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

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))

    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    # We want to visualize the output of the spatial transformers layer
    # after the training, we visualize a batch of input images and
    # the corresponding transformed batch using STN.

    def visualize_stn():
        with torch.no_grad():
            # Get a batch of training data
            data = next(iter(test_loader))[0].to(device)

            input_tensor = data.cpu()
            transformed_input_tensor = model.stn(data).cpu()

            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')

    for epoch in range(1, 1 + 1):
        train(epoch)
        test()

    # Visualize the STN transformation on some input batch
    visualize_stn()

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
