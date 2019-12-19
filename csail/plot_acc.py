
import sys
import matplotlib.pyplot as plt
import torch

def read_acc(checkpointFilePath):
    checkpoint = torch.load(checkpointFilePath)
    acc_arr = checkpoint['acc_arr']
    return acc_arr

def plot(acc_arr):
    epoch = list(range(len(acc_arr)))
    accuracy = list(map(lambda x: 100*x, acc_arr))

    fig, ax = plt.subplots()
    ax.plot(epoch, accuracy)
    ax.set(xlabel='Epoch', ylabel='Avg. Accuracy (%)', title='Classification Epoch vs. Accuracy')
    ax.grid()

    fig.savefig('epoch_accuracy.png')
    plt.show()

if __name__ == "__main__":

    checkpointFilePath = sys.argv[1]
    acc_arr = read_acc(checkpointFilePath)
    plot(acc_arr)