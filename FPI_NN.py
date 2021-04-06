"""
Demo program with FPI_NN for multi-label classificaiton.

Ref: Younghan Jeon, Minsik Lee, and Jin Young Choi,
    "Differentiable Forward and Backward Fixed-point Iteration Layers,"
    IEEE Access, January 22, 2021.
License: GPLv3

Copyright (C) 2021 Younghan Jeon, Minsik Lee
This file is part of PyTorch-fpilayer.

PyTorch-fpilayer is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

PyTorch-fpilayer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PyTorch-fpilayer. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import fixed_point_iteration as fp

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0")

trainX, trainY = np.split(np.loadtxt(os.path.join("data/bibtex_train_final.csv"), delimiter=',', dtype=np.float32), [1836], axis=1)
testX, testY = np.split(np.loadtxt(os.path.join("data/bibtex_test_final.csv"), delimiter=',', dtype=np.float32), [1836], axis=1)

nTrain = trainX.shape[0]
nTest = testX.shape[0]
nFeatures = trainX.shape[1]
nLabels = trainY.shape[1]

num_epoch = 3000

batch_size = 40
test_batch = 2515

learning_rate = 1e-3

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(1836+159, 600)
        self.fc2 = nn.Linear(600, 159)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

class G(nn.Module):
    def __init__(self, model):
        super(G, self).__init__()
        self.model = model

    def forward(self, x, y):
        X, = x
        inps, = y
        out = self.model(torch.cat((inps, X), 1))

        return out,

def F1_score(trueY, predY):

    predY_bin = torch.round(predY)
    trueY_bin = trueY
    I = torch.min(predY_bin, trueY_bin)
    U = torch.max(predY_bin, trueY_bin)

    return 2 * I.sum(1) / (I.sum(1) + U.sum(1))

def term(cnt, x, px):
    return fp.default_term_cond(cnt, x, px, threshold=1e-8, normalized=True, display=False, maxIter=4000)

g = G(BaseNet())
g.to(device)
g = nn.DataParallel(g)

criterion = nn.BCELoss().cuda()
optimizer = torch.optim.Adam(g.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400])

best_F1 = 0.
best_epoch = 0
for epoch in range(1, num_epoch + 1):

    rand = list(range(nTrain))
    np.random.shuffle(rand)

    train_loss = 0.

    for i in range(int(math.ceil(nTrain/float(batch_size)))):

        inputs = torch.from_numpy(trainX[rand[batch_size * i:batch_size * (i + 1)]]).float().cuda()
        targets = torch.from_numpy(trainY[rand[batch_size * i:batch_size * (i + 1)]]).float().cuda()

        """ compute x_hat """
        x_init = torch.zeros((batch_size, nLabels), requires_grad=True).cuda()
        fpi = fp.FixedPointIteration(g=g, forward_cond=term, backward_cond=term)
        x_hat, = fpi(y=[inputs], x0=[x_init])

        loss = criterion(x_hat, targets).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        temp = '%d epoch, iter %d/%d' % (epoch, i + 1, nTrain/batch_size)
        sys.stdout.write('\r' + temp)

    print(', train_loss: %f' % (train_loss/math.ceil(nTrain/batch_size)))

    g.eval()
    F1_sum = 0.
    for i in range(int(math.ceil(nTest / float(test_batch)))):

        last = min(test_batch * (i + 1), nTest)
        num = last - test_batch * i

        inputs = torch.from_numpy(testX[test_batch * i:last]).float().cuda()

        """ compute x_hat """
        x_init = torch.zeros((num, nLabels), requires_grad=True).cuda()
        fpi = fp.FixedPointIteration(g=g, forward_cond=term, backward_cond=term)
        x_hat, = fpi(y=[inputs], x0=[x_init])

        targets = torch.from_numpy(testY[test_batch * i:last]).float().cuda()

        F1_sum += F1_score(targets, x_hat).sum().item()

    F1 = F1_sum / nTest
    print('F1 score: %f\n' % F1)
    g.train()

    if F1 > best_F1:
        best_F1 = F1
        best_epoch = epoch
        if F1 > 0.4:
            torch.save(g, 'model/' + str(epoch) + 'ep-nn ' + str(round(F1 * 10000) / 10000))

    print('best_F1: %f (epoch %d)\n' % (best_F1, best_epoch))
