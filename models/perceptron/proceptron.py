import torch
import torch.nn as nn


class Perceptron(object):
    def __init__(self, feature_size, data, lr=0.1):
        self.data = data
        self.lr = lr

        self.weight = torch.ones((feature_size,))
        self.bia = torch.Tensor([0])

    def train(self):

        while True:
            flag = True

            for point, y in self.data:
                y_ = torch.dot(self.weight, point) + self.bia
                tmp = torch.mul(y_, y)
                tmp = torch.sign(tmp).item()

                if tmp <= 0:
                    self.weight = self.weight + self.lr * y * point
                    self.bia = self.bia + self.lr * y

                    flag = False

            if flag:
                break


if __name__ == '__main__':
    perceptron = Perceptron(2,
                            [(torch.Tensor([1, 2]), torch.Tensor([-1])),
                             (torch.Tensor([2, 2]), torch.Tensor([-1])),
                             (torch.Tensor([2, 0]), torch.Tensor([-1])),
                             (torch.Tensor([0, 0]), torch.Tensor([1])),
                             (torch.Tensor([1, 0]), torch.Tensor([1])),
                             (torch.Tensor([0, 1]), torch.Tensor([1]))], )

    perceptron.train()
