import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        
        self.hidden1 = nn.Linear(self.board_x, self.args.num_channels)
        
        self.hidden2 = nn.Linear(self.args.num_channels, self.args.num_channels)


        # self.conv1 = nn.Conv1d(1, args.num_channels, 2, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(args.num_channels, args.num_channels, 2, stride=1, padding=1)
        self.conv3 = nn.Conv1d(self.args.num_channels, args.num_channels, kernel_size=1, stride=1)
        # self.conv4 = nn.Conv1d(args.num_channels, args.num_channels, 2, stride=1)

        self.bn1 = nn.BatchNorm1d(self.args.num_channels)
        self.bn2 = nn.BatchNorm1d(self.args.num_channels)
        self.bn3 = nn.BatchNorm1d(args.num_channels)
        self.bn4 = nn.BatchNorm1d(args.num_channels)

        # self.fc1 = nn.Linear(args.num_channels*(self.board_x-4), 2048)
        # self.fc_bn1 = nn.BatchNorm1d(2048)

        # self.fc2 = nn.Linear(1024, 512)
        # self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(self.args.num_channels, self.action_size)

        self.fc4 = nn.Linear(self.args.num_channels, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x 
        s = s.view(1, self.board_x)                # batch_size x 1 x board_x
        s = F.relu(self.bn1(self.hidden1(s)))
        s = F.relu(self.bn2(self.hidden2(s)))


        # s = F.relu(self.bn3(self.conv3(s)))
        # s = F.relu(self.bn4(self.conv4(s)))



        # s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x
        # print(s.shape)
        # s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x 
        # print(s.shape)
        # s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        # s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        # s = s.view(-1, self.args.num_channels*(self.board_x))
        # s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        # s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class ConnectTwoConvNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(ConnectTwoConvNNet, self).__init__()
        
        self.hidden1 = nn.Linear(self.board_x, self.args.num_channels)
        
        self.hidden2 = nn.Linear(self.args.num_channels, self.args.num_channels)


        # Define a 1d convolution on an input of shape [1,args.num_channels]
        self.conv3 = nn.Conv1d(1, self.args.num_channels, 1, stride=1, padding=0)

        # self.conv1 = nn.Conv1d(1, args.num_channels, 2, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(args.num_channels, args.num_channels, 2, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(self.args.num_channels, args.num_channels, kernel_size=1, stride=1)
        # self.conv4 = nn.Conv1d(args.num_channels, args.num_channels, 2, stride=1)



        self.bn1 = nn.BatchNorm1d(args.num_channels)
        self.bn2 = nn.BatchNorm1d(args.num_channels)
        self.bn3 = nn.BatchNorm1d(args.num_channels)
        self.bn4 = nn.BatchNorm1d(args.num_channels)

        # self.fc1 = nn.Linear(args.num_channels*(self.board_x-4), 2048)
        # self.fc_bn1 = nn.BatchNorm1d(2048)

        # self.fc2 = nn.Linear(1024, 512)
        # self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(args.num_channels, self.action_size)

        self.fc4 = nn.Linear(args.num_channels, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x 
        s = s.view(1, self.board_x)                # batch_size x 1 x board_x
        s = F.relu(self.bn1(self.hidden1(s)))
        s = F.relu(self.bn2(self.hidden2(s)))

        # Perform a 1d convolution for an input of size [1,512]
        s = F.relu(self.bn3(self.conv3(s)))

        # s = F.relu(self.bn3(self.conv3(s)))
        # s = F.relu(self.bn4(self.conv4(s)))



        # s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x
        # print(s.shape)
        # s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x 
        # print(s.shape)
        # s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        # s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        # s = s.view(-1, self.args.num_channels*(self.board_x))
        # s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        # s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)