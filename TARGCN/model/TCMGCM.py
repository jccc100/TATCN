import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
# from model.GRU import GRU

from model.TA import TA_layer
from model.MGCN import MGCN as GCN
from torch.autograd import Variable
import math
device=torch.device('cuda')

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x) # 残差
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # self.gcn=GCN(dim_in, dim_out, adj, cheb_k, embed_dim)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        # b, t, n, d = x.shape
        # out1=self.network[0](x.permute(0, 2, 3, 1).reshape(b * n, d, t))
        # out1=self.gcn(out1.reshape(b, n, d, t).permute(0, 3, 1, 2),node_embeddings) # btnd
        # out2=self.network[1](out1.permute(0, 2, 3, 1).reshape(b * n, d, t))
        # out2 = self.gcn(out2.reshape(b, n, d, t).permute(0, 3, 1, 2),node_embeddings)  # btnd
        # out3 = self.network[1](out2.permute(0, 2, 3, 1).reshape(b * n, d, t))
        # out3 = self.gcn(out3.reshape(b, n, d, t).permute(0, 3, 1, 2),node_embeddings)  # btnd
        # return out3
        return self.network(x)


class TCNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel=3, dropout=0.5):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv2 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv3 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.bn = nn.BatchNorm2d(out_features)
        self.dropout = dropout

    def forward(self, inputs):
        """
        param inputs: (batch_size, timestamp, num_node, in_features)
        return: (batch_size, timestamp - 2, num_node, out_features)
        """
        inputs = inputs.permute(0, 3, 2, 1)  # (btnf->bfnt)
        out = torch.relu(self.conv1(inputs)) * \
            torch.sigmoid(self.conv2(inputs))
        #out = torch.relu(out + self.conv3(inputs))
        out = self.bn(out)
        out = out.permute(0, 3, 2, 1)
        out = torch.dropout(out, p=self.dropout, train=self.training)
        return out


class TCN(nn.Module):
    def __init__(self, n_history, in_features, mid_features) -> None:
        super(TCN, self).__init__()
        # odd time seriers: number of layer is n_hitory // 2
        # even time seriers: number of layer is n_history//2-1 + a single conv layer.
        # -> Aggregate information from time seriers to one unit
        assert(n_history >= 3)
        self.is_even = False if n_history % 2 != 0 else True

        self.n_layers = n_history // \
            2 if n_history % 2 != 0 else (n_history // 2 - 1)

        self.tcn_layers = nn.ModuleList([TCNLayer(in_features, mid_features)])
        for i in range(self.n_layers - 1):
            self.tcn_layers.append(TCNLayer(mid_features, mid_features))

        if self.is_even:
            self.tcn_layers.append(
                TCNLayer(mid_features, mid_features, kernel=2))

        self.upsample = None if in_features == mid_features else nn.Conv2d(
            in_features, mid_features, kernel_size=1)

    def forward(self, inputs):
        out = self.tcn_layers[0](inputs)
        if self.upsample:
            ResConn = self.upsample(inputs.permute(0, 3, 2, 1))
            ResConn = ResConn.permute(0, 3, 2, 1)
        else:
            ResConn = inputs

        out = out + ResConn[:, 2:, ...]

        for i in range(1, self.n_layers):
            out = self.tcn_layers[i](
                out) + out[:, 2:, ...] + ResConn[:, 2 * (i+1):, ...]

        if self.is_even:
            out = self.tcn_layers[-1](out) + out[:, -1,
                                                 :, :].unsqueeze(1) + ResConn[:, -1:, ...]

        return out


class Encoder(nn.Module):
    def __init__(self, n_history, n_predict, in_features, mid_features,out_features) -> None:
        super(Encoder, self).__init__()
        assert(n_history >= 3)
        self.n_predict = n_predict

        self.tcn = TCN(n_history=n_history,
                       in_features=in_features, mid_features=mid_features)

        self.fully = nn.Linear(mid_features, mid_features)
        self.out_linear = nn.Linear(1, n_predict)
        self.weight = nn.Parameter(
            torch.FloatTensor(mid_features, out_features))
        self.bais = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameter()



    def reset_parameter(self):
        for param in self.fully.parameters():
            param.data.normal_()

    def forward(self, inputs):
        out = self.tcn(inputs) # btnc

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = torch.relu(self.fully(out))
        out = out.reshape(out.shape[0], out.shape[1], 1, -1)
        out = out.permute(0, 2, 1, 3)
        out=self.out_linear(out.permute(0,3,2,1))
        out=torch.relu(torch.matmul(out.permute(0,3,2,1), self.weight) + self.bais)
        return out

class TCMGCN_cell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj,tcn_in,tcn_mid,tcn_out,num_layers=1):
        super(TCMGCN_cell, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.adj=adj
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        # self.dcrnn_cells = nn.ModuleList()
        # self.dcrnn_cells.append(GRU(node_num, dim_in, dim_out, self.adj,cheb_k, embed_dim))
        # self.tcn=TemporalConvNet(dim_in,[3],3,0.2)
        # for _ in range(1, num_layers):
        #     self.dcrnn_cells.append(GRU(node_num, dim_out, dim_out,self.adj ,cheb_k, embed_dim))

        self.gcn=GCN(dim_in, dim_out, self.adj, cheb_k, embed_dim)
        self.tcn = Encoder(12,12,tcn_in,tcn_mid,tcn_out)
        self.TA_layer = TA_layer(dim_out, dim_out, 2, 2)

    def forward(self, x, node_embeddings1,node_embeddings2):
        # print("aaaaaaaaaaaaa",x.shape)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        b, t, n, d = x.shape
        x = x.to(device=device)
        input = x
        # tcn_input = x  # b*n d t
        # tcn_input = x

        TA_output = self.TA_layer(input)
        tcn_output = self.tcn(input)
        x_gconv_TA = self.gcn(TA_output, node_embeddings1,node_embeddings2)
        # x_gconv_TA = self.gcn(x_gconv_TA, node_embeddings1,node_embeddings2)

        x_gconv_tcn = self.gcn(tcn_output, node_embeddings1,node_embeddings2)
        # x_gconv_tcn = self.gcn(x_gconv_tcn, node_embeddings1,node_embeddings2)
        # print("aaaaaaaaaaaaa", (x_gconv_tcn+x_gconv_TA).shape)
        return x_gconv_tcn+x_gconv_TA#,node_embeddings1,node_embeddings2
        # return x_gconv_TA
        # return self.gcn(self.TA_layer(TA_output), node_embeddings1,node_embeddings2)+x_gconv_TA


        # TA_output = self.TA_layer(x_gconv_tcn+x_gconv_TA)
        # tcn_output = self.tcn(x_gconv_tcn+x_gconv_TA)
        # return TA_output+tcn_output


        # current_inputs = x
        # output_hidden = []
        # for i in range(self.num_layers):
        #     state = init_state[i]
        #     inner_states = []
        #     for t in range(seq_length):
        #         state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
        #         inner_states.append(state)
        #     output_hidden.append(state)
        #     current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        # current_inputs=self.TA_layer(current_inputs)
        # return current_inputs, output_hidden

        # return tcn_output

    # def init_hidden(self, batch_size):
    #     init_states = []
    #     for i in range(self.num_layers):
    #         init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
    #     return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

# class TARGCN_cell2(nn.Module):
#     def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj,num_layers=1):
#         super(TARGCN_cell2, self).__init__()
#         assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
#         self.adj=adj
#         self.node_num = node_num
#         self.input_dim = dim_in
#         self.num_layers = num_layers
#
#         # self.dcrnn_cells = nn.ModuleList()
#         # self.dcrnn_cells.append(GRU(node_num, dim_in, dim_out, self.adj,cheb_k, embed_dim))
#         # self.tcn=TemporalConvNet(dim_in,[3],3,0.2)
#         # for _ in range(1, num_layers):
#         #     self.dcrnn_cells.append(GRU(node_num, dim_out, dim_out,self.adj ,cheb_k, embed_dim))
#
#         self.gcn=GCN(dim_in, dim_out, self.adj, cheb_k, embed_dim)
#         self.tcn = TemporalConvNet( dim_in, [1, 1, 1], 2, 0.2)
#         self.TA_layer = TA_layer(dim_out, dim_out, 2, 2)
#
#     def forward(self, x, node_embeddings):
#
#         assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
#         seq_length = x.shape[1]
#         b, t, n, d = x.shape
#         x = x.to(device=device)
#         input = self.gcn(x,node_embeddings)
#         # tcn_input = x  # b*n d t
#         # tcn_input = x
#
#
#         TA_output = self.TA_layer(input)
#         #tcn_output = self.tcn(input.permute(0, 2, 3, 1).reshape(b * n, d, t)).reshape(b, n, d, t).permute(0, 3, 1, 2)
#         # x_gconv_TA = self.gcn(TA_output, node_embeddings)
#         # x_gconv_TA = self.gcn(x_gconv_TA, node_embeddings)
#
#         # x_gconv_tcn = self.gcn(tcn_output, node_embeddings)
#         # x_gconv_tcn = self.gcn(x_gconv_tcn, node_embeddings)
#
#
#
#         # current_inputs = x
#         # output_hidden = []
#         # for i in range(self.num_layers):
#         #     state = init_state[i]
#         #     inner_states = []
#         #     for t in range(seq_length):
#         #         state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
#         #         inner_states.append(state)
#         #     output_hidden.append(state)
#         #     current_inputs = torch.stack(inner_states, dim=1)
#         #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
#         #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
#         #last_state: (B, N, hidden_dim)
#         # current_inputs=self.TA_layer(current_inputs)
#         # return current_inputs, output_hidden
#         # return x_gconv_tcn+x_gconv_TA
#         return TA_output+tcn_output
#         # return tcn_output
#
#     # def init_hidden(self, batch_size):
#     #     init_states = []
#     #     for i in range(self.num_layers):
#     #         init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
#     #     return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class TCMGCN(nn.Module):
    def __init__(self, args,adj=None):
        super(TCMGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.adj=adj
        # self.default_graph = args.default_graph

        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = TCMGCN_cell(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim,self.adj, 1,64,1,args.num_layers)
        # self.encoder2 = TCMGCN_cell(args.num_nodes, args.rnn_units, args.rnn_units, args.cheb_k,
        #                             args.embed_dim, self.adj, 64, 64, 1, args.num_layers)
        self.encoder2 = TCMGCN_cell(args.num_nodes, args.rnn_units, args.rnn_units, args.cheb_k,
                                   args.embed_dim, self.adj, 64,64,1,args.num_layers)
        # self.encoder3 = TCMGCN_cell(args.num_nodes, args.rnn_units, args.rnn_units, args.cheb_k,
        #                             args.embed_dim, self.adj, 64, 64, 1, args.num_layers)

        self.end_conv = nn.Conv2d(6, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.FC = nn.Linear(6,6)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        # init_state = self.encoder.init_hidden(source.shape[0])
        output= self.encoder(source, self.node_embeddings1,self.node_embeddings2)      #B, T, N, hidden
        output= self.encoder2(output, self.node_embeddings1,self.node_embeddings2)+output      #B, T, N, hidden
        # output= self.encoder3(output, self.node_embeddings1,self.node_embeddings2)+output      #B, T, N, hidden


        output = output[:, -6:, :, :]
        output = self.end_conv((output))                         #B, T*C, N, 1


        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) # b t c n
        output = output.permute(0, 1, 3, 2)                             #B, T(12), N, C

        # output2 = self.encoder2(output, self.node_embeddings1, self.node_embeddings2)  # B, T, N, hidden
        # output2 = output2[:, -6:, :, :]
        # output2 = self.end_conv((output2))  # B, T*C, N, 1
        #
        # output2 = output2.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)  # b t c n
        # output2 = output2.permute(0, 1, 3, 2)+output

        return output

if __name__=='__main__':



    import argparse
    import configparser
    config = configparser.ConfigParser()
    config_file = 'PEMSD8.conf'

    config.read(config_file)

    args = argparse.ArgumentParser(description='arguments')

    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
    # args.add_argument('--embed_dim', default=2, type=int)
    args = args.parse_args()

    num_node = args.num_nodes
    input_dim = args.input_dim
    hidden_dim = args.rnn_units
    output_dim = args.output_dim
    horizon = args.horizon
    num_layers = args.num_layers
    adj = torch.ones((num_node,num_node))
    # print(adj.shape)
    node_embeddings = nn.Parameter(torch.randn(num_node, 2), requires_grad=True)
    net = TARGCN(args,adj)
    net=net.cuda()

    # source: B, T_1, N, D
    # target: B, T_2, N, D
    x=torch.randn(32,12,170,1)
    tar=torch.randn(32,12,170,1)
    out=net(x,tar)

    print(out.shape)
