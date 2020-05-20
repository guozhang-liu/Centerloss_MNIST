import matplotlib.pyplot as plt
import torch
import numpy as np

def DrawPics(features, labels, epoch):
    # features, labels, epoch = features.data.cpu().numpy(), labels.data.cpu().numpy(), epoch.data.cpu().numpy()
    plt.clf()
    color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    for i in range(10):
        plt.plot(features[labels == i, 0], features[labels == i, 1], ".", c=color[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title("Epoch-{}".format(epoch))
    plt.savefig("Pics/Epoch-{}".format(epoch))


if __name__ == "__main__":
    data = torch.tensor([[3, 4], [5, 6], [7, 8], [9, 8], [6, 5]], dtype=torch.float32)
    label = torch.tensor([0, 0, 1, 0, 1], dtype=torch.float32)
    # DrawPics(data, label, 10000000)
    for i in range(2):
        a = data[label == i]
        print(a.shape)
