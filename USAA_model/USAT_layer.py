
#添加channels
import torch
import torch.nn as nn
import torch.nn.functional as F
from utlis import BinarizedF

class BatchGAT(nn.Module):
    def __init__(self, channels, n_heads, f_in, n_units, dropout, attn_dropout, bias =False):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units)
        self.f_in = f_in
        self.dropout = dropout
        self.bias = bias

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            f_in = n_units[i-1] * n_heads[i] if i else self.f_in
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(channels,n_heads[i], f_in=f_in,
                                             f_out=n_units[i], attn_dropout=attn_dropout,bias =self.bias)
            )

    def forward(self, x, adj):
        bs, chs, n = x.size()[:3]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj) # bs x c x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=2)
            else:
                x = F.elu(x.permute(0, 1, 3, 2, 4).contiguous().view(bs, chs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, channels,n_head, f_in, f_out, attn_dropout, bias = False):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = nn.Parameter(torch.Tensor(channels,n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(channels,n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(channels,n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)


        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        h = h.unsqueeze(2)
        h_prime = torch.matmul(h, self.w) # bs x c x n_head x n x f_out
        B, C, H, N, F = h.size()  # h is of size bs x c x n x f_in

        # #yanban
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src) # bs x c x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst) # bs x c x n_head x n x 1
        attn = attn_src.expand(-1, -1,-1, -1, N) + attn_dst.expand(-1, -1,-1, -1, N).permute(0, 1, 2,4, 3) # bs  x c x n_head x n x n
        attn_all = self.leaky_relu(attn)

          # # bs x c x n_head x n x f_out
        eye = torch.eye(N).expand((B,N,N)).to(adj.device)
        adj = adj + eye
        adj = adj.unsqueeze(1).repeat(1,C,1,1) # bs x n x n -> bs x c x n x n
        mask = (adj.unsqueeze(2)==0).repeat(1,1,H,1,1) # bs x c x 1 x n x n
        attn_all.data.masked_fill_(mask, -1e12)
        attn_all = self.softmax(attn_all)  # bs x c x n_head x n x n   #应该先置ttn.data.masked_fill_(mask, -1e12),后softmax. 但矩阵稀疏，这里调整

        attn_all = self.dropout(attn_all)
        output = torch.matmul(attn_all, h_prime) # bs x c x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class OP_Attention(nn.Module):
    def __init__(self,channels, f_in):
        super(OP_Attention,self).__init__()
        self.op = nn.Parameter(torch.Tensor(channels, 2 * f_in, 1))

        self.op_S = nn.Parameter(torch.Tensor(channels,f_in * 2, 1))
        self.op_Q = nn.Parameter(torch.Tensor(channels,f_in * 2, 1))

        self.sigmoid = nn.Sigmoid()
        self.bn2d = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.op)
        nn.init.xavier_uniform_(self.op_Q)
        nn.init.xavier_uniform_(self.op_S)

    def forward(self,h):
        B, C, N, F = h.size()
        h_src = torch.repeat_interleave(h[:,:,0,:].unsqueeze(dim=2), N, dim=2)
        h_op = torch.cat((h_src, h), dim=-1)
        attn_op = torch.matmul(h_op, self.op)
        attn_op = self.sigmoid(attn_op)
        # op_mask = BinarizedF.apply(attn_op)
        ones = torch.ones_like(attn_op)
        zeros = torch.zeros_like(attn_op)
        op_mask = torch.where(attn_op >= 0.5, ones, zeros)
        supp = torch.mul(h, op_mask)
        query = torch.mul(h, 1 - op_mask)
        supp_mean = torch.mean(supp, dim=2,keepdim=True)
        query_mean = torch.mean(query, dim=2,keepdim=True)
        source = h[:,:, 0, :].unsqueeze(2)
        supp_attn = torch.matmul(torch.cat([supp_mean, source], dim=-1), self.op_S)
        query_attn = torch.matmul(torch.cat([query_mean, source], dim=-1), self.op_Q)
        op_attn = torch.cat([supp_attn, query_attn], dim=-1)
        op_attn = self.softmax(op_attn)
        h_opn = torch.mul(supp,op_attn[:,:,:,0].unsqueeze(3)) +torch.mul(query,op_attn[:,:,:,1].unsqueeze(3))
        h_opn = self.bn2d(h_opn)
        h_opn = self.elu(h_opn)
        return h_opn

class GAT_block(nn.Module):
    def __init__(self,channels, num_heads, f_in, h_DUGAT, dropout, attn_dropout):
        super(GAT_block, self).__init__()

        self.BMGAT = BatchGAT(channels, num_heads, f_in, h_DUGAT, dropout= dropout, attn_dropout=attn_dropout)
        self.bn2d = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()

    def forward(self, x, adj):
        x_agg = self.BMGAT(x, adj)
        mask = (x_agg==0)
        x_agg = self.bn2d(x_agg)
        x_agg = self.elu(x_agg)
        x_agg.data.masked_fill_(mask, 0.0)

        return x_agg

class LSTM_block(nn.Module):
    def __init__(self,channels, x_dim, h_dim, batch_first=True):
        super(LSTM_block, self).__init__()

        self.lstm = nn.LSTM(input_size=x_dim * 2, hidden_size = h_dim, batch_first= batch_first)
        self.bn1d = nn.BatchNorm1d(channels)
        self.elu = nn.ELU()

    def forward(self, x):

        x_g_s = x[:, :, 0, :]
        x_g_r = torch.mean(x[:, :, 1:, :], dim=2, keepdim=False)
        x_g_t = torch.cat((x_g_s, x_g_r), dim=-1)
        output, (h_n, c_n) = self.lstm(x_g_t)
        output = self.bn1d(output)
        output = self.elu(output)
        output_mean= torch.mean(output, dim=1)
        return output_mean
