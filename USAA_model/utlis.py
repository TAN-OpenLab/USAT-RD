import torch
from torch.autograd import Function,Variable
import torch.nn as nn
import torch.nn.functional as F



class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        a = torch.ones_like(input)
        b = torch.zeros_like(input)
        output = torch.where(input >= 0.5, a, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input_g =  0.5 - torch.abs((input-0.5))
        ones = torch.ones_like(input)
        zeros = -torch.ones_like(input)
        grad_output = torch.where(input >= 0.5, ones,zeros)
        return grad_output


# class BinarizedModule(nn.Module):
#     def __init__(self):
#         super(BinarizedModule, self).__init__()
#         self.BF = BinarizedF()
#     def forward(self,input):
#         #print(input.shape)
#         output = self.BF(input)
#         return output

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def evaluationclass(prediction, y):  # 2 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]

        ## for class 1
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        ## for class 2
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1

    ## print result
    Acc_all = round(float(TP1 + TP2) / float(len(y) ), 4)
    Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
    if (TP1 + FP1)==0:
        Prec1 =0
    else:
        Prec1 = round(float(TP1) / float(TP1 + FP1), 4)
    if (TP1 + FN1 )==0:
        Recll1 =0
    else:
        Recll1 = round(float(TP1) / float(TP1 + FN1 ), 4)
    if (Prec1 + Recll1 )==0:
        F1 =0
    else:
        F1 = round(2 * Prec1 * Recll1 / (Prec1 + Recll1 ), 4)

    Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
    if (TP2 + FP2)==0:
        Prec2 =0
    else:
        Prec2 = round(float(TP2) / float(TP2 + FP2), 4)
    if (TP2 + FN2 )==0:
        Recll2 =0
    else:
        Recll2 = round(float(TP2) / float(TP2 + FN2 ), 4)
    if (Prec2 + Recll2 )==0:
        F2 =0
    else:
        F2 = round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)

    return  Acc_all,Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2