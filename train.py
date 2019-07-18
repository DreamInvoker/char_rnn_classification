from data import get_data
from model import RNN
import torch
from torch import optim
import random
from torch import nn
import time
import math
import matplotlib.pyplot as plt

def time_since(since):
    t = time.time() - since
    m = math.floor(t / 60)
    t -= m * 60
    return '%dm%ds' % (m, t)


def fit(model, epochs, n_hidden, criterion, x_train, y_train, x_val, y_val, opt, prefix, dev=torch.device('cpu')):

    history_loss = []
    for epoch in range(epochs):
        hidden = torch.zeros((1, n_hidden)).to(dev)
        loss = 0
        tic = time.time()
        index = list(range(len(x_train)))
        random.shuffle(index)
        model.train()

        opt.zero_grad()
        for i in index:
            x, y = x_train[i].to(dev), y_train[i].to(dev)
            for l in x:
                pred, hidden = model(l, hidden)
            loss += criterion(pred, y)
        loss /= len(x_train)
        loss.backward()
        opt.step()

        print('Epoch {}'.format(epoch))
        print('-'*50)
        print('train-time: {}\nloss: {}'.format(time_since(tic), loss))
        history_loss.append(loss)
        tic = time.time()

        model.eval()
        total = len(x_val)
        count = 0
        with torch.no_grad():
            for i in range(len(x_val)):
                x, y = x_val[i].to(dev), y_val[i].to(dev)
                for l in x:
                    pred, hidden = model(l, hidden)
                pred = torch.argmax(pred)
                if pred == y.item():
                    count += 1

        print('validation-acc:%.2d%%' % (count / total * 100))
        print('validation-time: {}'.format(time_since(tic)))
        print('-'*50)

        torch.save({
            'validation-acc': count / total * 100,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'epoch': epoch,
            'loss': loss
        }, 'weights/%s-%03d.pth' % (prefix, epoch))
    return history_loss


if __name__ == '__main__':
    n_hidden = 128
    lr = 0.3
    momentum = 0.9
    epochs = 10
    dev = torch.device('cpu')
    all_letter, all_categories, x_train, y_train, x_val, y_val = get_data()

    rnn = RNN(len(all_letter), n_hidden, len(all_categories))

    # rnn = nn.DataParallel(rnn.to(dev), device_ids=[1, 0, 2, 3])

    opt = optim.SGD(rnn.parameters(), lr=lr, momentum=momentum)

    history_loss = fit(rnn, epochs, n_hidden, nn.CrossEntropyLoss(), x_train, y_train, x_val, y_val, opt, 'SRN/one', dev)

    plt.figure()
    plt.plot(history_loss, list(range(epochs)))
    plt.show()

