from random import shuffle
import numpy as np
import os
import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        pid = os.getpid()
        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        len_dataset = len(train_loader.dataset)
        batch_size  = train_loader.batch_size

        for epoch in range(1, num_epochs+1):
            #for i in range(iter_per_epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                
                data, target = Variable(data), Variable(target)
                data = data.cuda()
                target = target.cuda()
                model.train()
                optim.zero_grad()
                output = model(data)
            
                loss = self.loss_func(output, target)

                loss.backward()
                optim.step()
                self.train_loss_history.append(loss)
                if batch_idx % 1000 == 0:
                    print(batch_idx)
                    print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            model.eval()
            test_loss = 0
            correct = 0

            for data, target in train_loader:
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                pred = output.max(1)[1]
                test_loss += self.loss_func(output, target).item() # sum up batch loss
                correct +=pred.eq(target).sum().item()
            
            print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))
            self.train_acc_history.append(100. * correct / len(train_loader.dataset))   
            
            
            model.eval()
            test_loss = 0
            correct = 0
            #with torch.no_grad():
            for data, target in val_loader:
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                test_loss += self.loss_func(output, target).item() # sum up batch loss
                pred = output.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target).sum().item()

            test_loss /= len(val_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(val_loader.dataset),
                100. * correct / len(val_loader.dataset)))
            self.val_acc_history.append(100. * correct / len(val_loader.dataset))




        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
