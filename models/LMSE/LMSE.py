import torch


class LMSE(object):
    def __init__(self, data):
        self.data = data
        self.bias = torch.ones((data.shape[0], 1))

    def train(self):
        tmp = torch.mm(self.data.transpose(0, 1), self.data)
        tmp = torch.inverse(tmp)
        tmp = torch.mm(tmp, self.data.transpose(0, 1))
        tmp = torch.mm(tmp, self.bias)

        print(tmp)


if __name__ == '__main__':
    lmse = LMSE(torch.Tensor([[1, 2, -1],
                              [2, 2, -1],
                              [2, 0, -1],
                              [0, 0, 1],
                              [1, 0, 1],
                              [0, 1, 1]]))
    lmse.train()
