import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataloader import MyDataset, DataLoaderX
from MCD_model import Generator,Classifier_1
from utlis import LambdaLR, FocalLoss,Regularization
import os, sys
from sklearn.metrics import precision_score,recall_score,f1_score
import numpy as np
import random


class USSA_model(object):
    def __init__(self, args, device):
        # parameters
        self.dataset =args.dataset
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.batch_size = args.batch_size
        self.f_in = args.input_dim_G
        self.h_DUGAT = args.h_DUGAT
        self.channels = args.channels
        self.num_heads = args.num_heads
        self.num_nodes = args.num_nodes
        self.h_op = args.h_op
        self.h_UDGAT = args.h_UDGAT
        self.hidden_LSTM = args.hidden_LSTM
        self.dense_C = args.dense_C
        self.num_worker = args.num_worker
        self.save_dir = args.save_dir
        self.model_dir = args.model_dir
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.device = device
        self.b1, self.b2 = args.b1, args.b2

        # load classifer
        self.generator = Generator(self.channels,self.num_nodes, self.num_heads,self.f_in,
                                   self.h_DUGAT,self.h_op, self.h_UDGAT,self.hidden_LSTM)
        self.classifier_1 = Classifier_1(self.hidden_LSTM, self.dense_C)
        self.generator.to(self.device)
        self.classifier_1.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = FocalLoss(class_num =2, alpha= 0.4).to(self.device)
        #self.criterion = nn.MSELoss()
        #self.criterion = nn.BCELoss()
        #self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        #self.optimizer_C1 = torch.optim.Adam(self.classifier_1.parameters(), lr= self.lr, betas=(self.b1, self.b2))
        self.optimizer_C1 = torch.optim.SGD(self.classifier_1.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_C1,
                                                              lr_lambda=LambdaLR(self.epochs, self.start_epoch,
                                                                               decay_start_epoch=10).step)


    def train(self, datapath, start_epoch, ispath):
        self.train_hist = {}
        self.train_hist['train_loss'] =[]
        self.train_hist['test_loss'] = []
        self.train_hist['train_acc'] = []
        self.train_hist['pre'] =[]
        self.train_hist['recall'] = []
        self.train_hist['f1'] = []

        for epoch in range(start_epoch, self.epochs):
            self.classifier_1.train()
            train_loss_value = 0
            acc_value =0
            test_loss, pre, recall, f1 = 0, 0, 0, 0
            if ispath:
                filelist = os.listdir(os.path.join(datapath,'train_data'))
                idx = [i for i in range(len(filelist))]
                random.shuffle(idx)
                filelist = [filelist[i] for i in idx]

                for f in filelist:
                    train_dataset = MyDataset(os.path.join(datapath,'train_data', f))
                    trainratio = np.bincount(train_dataset.y.numpy())
                    classcount = trainratio.tolist()
                    train_weights = 1. / torch.tensor(classcount, dtype=torch.float)
                    self.weights = train_weights[train_dataset.y]
                    train_loader = DataLoaderX(train_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=False,
                                              sampler=WeightedRandomSampler(self.weights, len(self.weights)),
                                              num_workers=self.num_worker,
                                              drop_last=True,
                                              pin_memory=False)
                    train_b_l , acc_b_v = self.train_batch(epoch,train_loader)
                    train_loss_value += train_b_l
                    acc_value += acc_b_v
                train_loss_value = train_loss_value/len(filelist)
                acc_value = acc_b_v / len(filelist)
                print('train_loss_value:%.8f acc_value:%.8f' %(train_loss_value,acc_value))

                #test
                with torch.no_grad():
                    filelist = os.listdir(os.path.join(datapath, 'test_data'))
                    for f in filelist:
                        print(f)
                        test_dataset = MyDataset(os.path.join(datapath, 'test_data', f))

                        test_loader = DataLoaderX(test_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_worker,
                                                 drop_last=True,
                                                 pin_memory=False)
                        test_b_v, pre_b, recall_b, f1_b = self.eval(epoch, test_loader)
                        test_loss += test_b_v
                        pre += pre_b
                        recall += recall_b
                        f1 += f1_b
                    test_loss = test_loss / len(filelist)
                    pre = pre / len(filelist)
                    recall = recall /len(filelist)
                    f1 = f1 / len(filelist)

            else:
                dataset = MyDataset(datapath)
                train_size = int(0.8 * len(dataset))
                train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                            [train_size, len(dataset) - train_size])

                # Train data
                trainratio = np.bincount(dataset.y[train_dataset.indices].numpy())
                classcount = trainratio.tolist()
                train_weights = 1. / torch.tensor(classcount, dtype=torch.float)
                self.weights = train_weights[dataset.y[train_dataset.indices]]
                train_loader = DataLoader(train_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          sampler=WeightedRandomSampler(self.weights, len(self.weights)),
                                          num_workers=self.num_worker,
                                          drop_last=True,
                                          pin_memory=False)

                test_loader = DataLoader(dataset=test_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=self.num_worker,
                                         drop_last=True,
                                         pin_memory=False)

                train_loss_value, acc_value = self.train_batch(epoch, train_loader)
                with torch.no_grad():
                    test_loss, pre, recall, f1 = self.eval(epoch, test_loader)

            self.train_hist['train_loss'].append(train_loss_value)
            self.train_hist['train_acc'].append(acc_value)
            self.train_hist['test_loss'].append(test_loss)
            self.train_hist['pre'].append(pre)
            self.train_hist['recall'].append(recall)
            self.train_hist['f1'].append(f1)

            if epoch % 20 == 0:
                torch.save(self.classifier_1.state_dict(),
                           os.path.join(self.save_dir, self.model_dir, self.dataset,
                                        str(epoch) + '_classifier.pkl'))
                with open(os.path.join(self.save_dir, self.model_dir, self.dataset, 'predict.txt'), 'w') as f:
                    hist = [str(k) + ':' + str(self.train_hist[k]) for k in self.train_hist.keys()]
                    f.write('\n'.join(hist) + '\n')
                print('save classifer : %d epoch' % epoch)


    def train_batch(self, epoch, dataloader):

        train_loss_value = 0
        acc_value = 0
        self.classifier_1.train()
        for iter, sample in enumerate(dataloader):
            x, A, y = sample
            x = x.to(self.device)
            A = A.to(self.device)
            y = y.to(self.device)

            h = self.generator(x, A)
            pred = self.classifier_1(h)

            loss = self.criterion(pred, y)

            pred = pred.data.max(1)[1]
            acc = (pred == y).sum() / len(y)
            train_loss_value += loss.item()
            acc_value += acc.item()

            self.optimizer_C1.zero_grad()
            loss.backward()
            self.optimizer_C1.step()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [model loss: %f] [model acc: %f] "
                % (
                    epoch,
                    self.epochs,
                    iter,
                    len(dataloader),
                    loss.item(),
                    acc.item()
                )
            )
        train_loss_value = train_loss_value /(iter+1)
        acc_value = acc_value/(iter+1)
        return train_loss_value, acc_value


    def eval(self, epoch, dataloader):

        pre_value, recall_value, f1_value, test_loss_value = 0, 0, 0, 0
        self.classifier_1.eval()
        for iter, sample in enumerate(dataloader):
            x, A, y = sample
            x = x.to(self.device)
            A = A.to(self.device)
            y = y.to(self.device)

            h = self.generator(x, A)
            pred = self.classifier_1(h)
            test_loss = self.criterion(pred, y)

            y_ =y.cpu()

            pred = pred.data.max(1)[1].cpu()
            test_loss_value += test_loss.item()
            pre = precision_score(pred, y_, average='binary', zero_division=0)
            recall = recall_score(pred, y_, average='binary', zero_division=0)
            f1 = f1_score(pred, y_, average='binary', zero_division=0)
            pre_value += pre
            recall_value += recall
            f1_value += f1

        test_loss_value = test_loss_value / (iter + 1)
        pre_value = pre_value / (iter + 1)
        recall_value = recall_value / (iter + 1)
        f1_value = f1_value / (iter + 1)
        print('test_loss:%0.8f, pre:%0.8f, recall:%0.8f, f1:%0.8f' % (test_loss_value, pre_value, recall_value, f1_value))
        return test_loss_value, pre_value, recall_value, f1_value


    def load(self,start_epoch):
        save_dir = os.path.join(self.save_dir, self.model_dir, self.dataset)


        self.classifier_1.load_state_dict(
            torch.load(os.path.join(save_dir, str(start_epoch) + '_classifier.pkl'), map_location=self.device,
                       encoding='utf-8'))
        start_epoch +=1

        return start_epoch
