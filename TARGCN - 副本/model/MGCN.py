import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


device=torch.device('cuda')
def sym_norm_Adj(W):
    W=W.to(device=torch.device('cpu'))
    assert W.shape[0] == W.shape[1]
    W=W.cpu().detach().numpy()
    N = W.shape[0]
    W = W + 0.5*np.identity(N)
    D = np.diag(1.0/np.sum(W, axis=1))

    sym_norm_Adj_matrix = np.dot(np.sqrt(D),W)

    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix,np.sqrt(D))

    return sym_norm_Adj_matrix # D^-0.5AD^-0.5
class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, num_node,c_in,c_out,dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        global device
        self.in_channels=c_in
        self.dropout = nn.Dropout(p=dropout)
        # self.vff = nn.Linear(c_out, c_out)
        # nn.init.kaiming_uniform_(self.vff.weight, nonlinearity="relu")
        # self.conv1 = nn.Conv2d(c_in, c_out, (1, 3), bias=True,padding=0,padding_mode='zeros')
        # self.conv2 = nn.Conv2d(c_out, c_out, (1, 3), bias=True,padding=0,padding_mode='zeros')

        self.Wq=nn.Linear(c_in,c_out,bias=False)
        # nn.init.kaiming_uniform_(self.Wq.weight, nonlinearity="relu")
        self.Wk=nn.Linear(c_in,c_out,bias=False)
        # nn.init.kaiming_uniform_(self.Wk.weight, nonlinearity="relu")
        self.Wv=nn.Linear(c_in,c_out,bias=False)
        # # nn.init.kaiming_uniform_(self.Wv.weight, nonlinearity="relu")
    def forward(self, x,adj,score_his=None):
        '''
        :param x: (batch_size,t, N, C)
        :return: (batch_size, t,N, C)
        '''
        # batch_size, num_of_vertices, in_channels = x.shape
        b,t,n,c=x.shape
        Q = x
        K = x
        V = x
        # Q=self.Wq(x.reshape(b*t*n,c)).reshape(b,t,n,c)
        # K=self.Wk(x.reshape(b*t*n,c)).reshape(b,t,n,c)
        # V=self.Wv(x.reshape(b*t*n,c)).reshape(b,t,n,c)
        # Q=self.conv1(x.permute(0,3,2,1)).permute(0,3,2,1)
        # K=self.conv2(x.permute(0,3,2,1)).permute(0,3,2,1)
        # # V=self.vff(x.permute(0,3,2,1)).permute(0,3,2,1)
        # V=x

        score = torch.matmul(Q, K.transpose(2, 3))
        score=F.softmax(score,dim=1)
        # score=torch.einsum('btnm,mc->btnc',score,adj)
        score=torch.einsum("btnm,btmc->btnc",score,V)
        return F.relu(score) # (b n n)

class MGCN(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(MGCN, self).__init__()

        self.cheb_k = cheb_k

        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        self.SA = Spatial_Attention_layer(adj.shape[0], dim_in, dim_out)
        self.linear=nn.Linear(dim_in, dim_out,bias=True)
        # self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        # self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.weights_pool = nn.Parameter(torch.FloatTensor(self.sym_norm_Adj_matrix.shape[0], dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(self.sym_norm_Adj_matrix.shape[0], dim_out))

        self.alpha=nn.Parameter(torch.FloatTensor([0.9]),requires_grad=True)
        self.beta=nn.Parameter(torch.FloatTensor([0.9]),requires_grad=True)
        self.gamma=nn.Parameter(torch.FloatTensor([0.1]),requires_grad=True)
    def forward(self, x, node_embeddings1,node_embeddings2):
        #x shaped[B,T, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, T,N, C]
        node_num = node_embeddings1.shape[0]
        supports = torch.relu(torch.tanh(torch.mm(node_embeddings1, node_embeddings2.transpose(0, 1))-
                                    torch.mm(node_embeddings2, node_embeddings1.transpose(0, 1)))) # N N

        supports = torch.eye(node_num).to(supports.device)+supports
        #static
        x_static = torch.einsum("nm,btmc->btnc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x)
        # x_static = torch.einsum("nm,btmc->btnc",torch.softmax(self.sym_norm_Adj_matrix,dim=-1),x_static)
        # x_static = self.linear(x_static) # btnc
        x_static = torch.relu(x_static)

        #spatial attention
        x_sa = torch.relu(self.SA(x, self.sym_norm_Adj_matrix))
        # x_sa = torch.relu(self.SA(x_sa, self.sym_norm_Adj_matrix))

        # weights = torch.einsum('nd,dio->nio', node_embeddings1, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        # bias = torch.matmul(node_embeddings1, self.bias_pool)#N, dim_out
        weights = torch.einsum('nd,dio->nio', supports, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(supports, self.bias_pool)#N, dim_out

        x_g = torch.einsum("nm,btmc->btnc", supports, x)


        x_gconv = torch.einsum('btni,nio->btno', x_g, weights) + bias     #b, N, dim_out

        x_gconv = torch.relu(x_gconv)
        # return x_gconv+torch.sigmoid(x_static)*x_static+x_sa
        return self.alpha*x_gconv+self.beta*x_sa+self.gamma*x_static
        # return x_gconv

class MGCN_noGate(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(MGCN_noGate, self).__init__()
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        supports = torch.eye(node_num).to(supports.device)+supports

        weights = torch.einsum('nd,dio->nio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)#N, dim_out

        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, cheb_k, N, dim_in

        x_gconv = torch.einsum('bni,nio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv

class MGCN_linear(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(MGCN_linear, self).__init__()
        self.cheb_k = cheb_k
        self.linear=nn.Linear(dim_in, dim_out,bias=True)
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) # N N
        supports=torch.eye(node_num).to(supports.device)+supports # 1+A
        x_g = torch.einsum("nm,bmc->bnc", supports, x)      #B, N, dim_in
        x_gconv=self.linear(x_g)
        return x_gconv
class MGCN_SA(nn.Module):
    def __init__(self, dim_in, dim_out, adj,cheb_k, embed_dim):
        super(MGCN_SA, self).__init__()
        self.cheb_k = cheb_k
        self.sym_norm_Adj_matrix = torch.from_numpy(sym_norm_Adj(adj)).to(torch.float32).to(torch.device('cuda'))
        self.sym_norm_Adj_matrix=F.softmax(self.sym_norm_Adj_matrix)
        self.SA=Spatial_Attention_layer(adj.shape[0],dim_in,dim_out)
    def forward(self, x, node_embeddings):
        x_sa = self.SA(x,self.sym_norm_Adj_matrix)
        x_sa=F.relu(x_sa)
        return x_sa


if __name__=="__main__":
    data=torch.randn((64,12,170,1)).to(device=device)
    print(data.device)
    e1=nn.Parameter(torch.FloatTensor(170, 2)).to(device=device)
    e2=nn.Parameter(torch.FloatTensor(170, 2)).to(device=device)
    print(e1.device)
    print(e2.device)
    adj=torch.randn((170,170)).to(device=device)
    print(adj.device)
    gcn=EmbGCN(1,1,adj,2,2)
    gcn=gcn.cuda()
    out=gcn(data,e1,e2)
    print(out.shape)
