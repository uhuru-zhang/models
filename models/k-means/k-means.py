import random
import pandas as pd
import numpy as np

import torch


class KMean(object):
    def __init__(self, data, n):
        self.data = data
        self.n = n

    def train(self):
        centroids = [self.data[i][0] for i in random.sample([j for j in range(len(self.data))], self.n)]

        last_centroids = torch.cat([centroid.unsqueeze(0) for centroid in centroids.copy()], dim=0)
        while True:
            point_clusters = [[] for _ in range(self.n)]
            label_clusters = [[] for _ in range(self.n)]
            for (point, label_) in self.data:
                i, l2 = self.get_nearest_centroid(point, centroids)
                point_clusters[i].append(point.unsqueeze(0))
                label_clusters[i].append(label_)

            for i, cluster in enumerate(point_clusters):
                cluster = torch.cat(cluster, dim=0)
                centroids[i] = torch.mean(cluster, dim=0)

            current_centroids = torch.cat([centroid.unsqueeze(0) for centroid in centroids.copy()], dim=0)

            for label_cluster in label_clusters:
                label_num = [0] * 10
                for label_ in label_cluster:
                    label_num[int(label_)] += 1
                print(label_num)

            print("===================")

            if torch.min(current_centroids == last_centroids).item() == 1:
                print(last_centroids)
                break
            last_centroids = torch.cat([centroid.unsqueeze(0) for centroid in centroids.copy()], dim=0)

    def get_nearest_centroid(self, point, centroids):
        l2s = [(i, torch.norm((point - centroid))) for i, centroid in enumerate(centroids)]
        return min(l2s, key=lambda x: x[1])


# 6,7,7,8,8,9,7,8,9,8,9

if __name__ == '__main__':
    data_file = "./ClusterSamples.csv"
    label_file = "./SampleLabels.csv"

    origin_data = torch.from_numpy(pd.read_csv(data_file, dtype=np.float, header=None).values)
    labels = torch.from_numpy(pd.read_csv(label_file, dtype=np.float, header=None).values)

    data = []
    for i, label in enumerate(labels):
        data.append((origin_data[i], label))

    kmean = KMean(data=data, n=10)
    kmean.train()

#     kmean = KMean(data=torch.Tensor(
#         [[0, 0],
#          [1, 0],
#          [0, 1],
#          [1, 1],
#          [2, 1],
#          [1, 2],
#          [2, 2],
#          [3, 1],
#          [6, 6],
#          [7, 6],
#          [7, 7],
#          [8, 6],
#          [8, 7],
#          [9, 7],
#          [7, 8],
#          [8, 8],
#          [9, 8],
#          [8, 9],
#          [9, 9]]
#     ), n=2)
#     kmean.train()
#
# ([6, 6], [7, 6], [7, 7], [8, 6], [8, 7], [9, 7], [7, 8], [8, 8], [9, 8], [8, 9], [9, 9])
# ([0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2], [3, 1])
