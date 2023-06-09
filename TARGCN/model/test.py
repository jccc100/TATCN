import torch
import torch.nn as nn
import numpy as np


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
    def __init__(self, n_history, n_predict, in_features, mid_features) -> None:
        super(Encoder, self).__init__()
        assert(n_history >= 3)
        self.n_predict = n_predict

        self.tcn = TCN(n_history=n_history,
                       in_features=in_features, mid_features=mid_features)

        self.fully = nn.Linear(mid_features, mid_features)
        self.out_linear = nn.Linear(1, n_predict)
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

        return out.permute(0,3,2,1)


class GCNCell(nn.Module):
    def __init__(self, in_features, out_features, cheb_k=3, dropout=0.5) -> None:
        super(GCNCell, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bais = nn.Parameter(torch.FloatTensor(out_features))

        self.cheb_k = cheb_k
        self.dropout = dropout

        self.reset_parameter()

    def reset_parameter(self):
        self.weight.data.normal_()
        self.bais.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, timestamp, num_node, num_features)
        adj: (num_node, num_node)
        """
        supports = [torch.eye(adj.shape[0]).to(adj.device), adj]
        for k in range(2, self.cheb_k):
            supports.append(torch.matmul(2 * adj, supports[-1]) - supports[-2])

        supports = torch.stack(supports, dim=0)

        lfs = torch.einsum('kij,jbtf->bitf', supports,
                           inputs.permute(2, 0, 1, 3))
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bais)
        return result.permute(0, 2, 1, 3)


class TransLayer(nn.Module):
    def __init__(self, num_node, n_history, n_predict, in_features, n_head, n_dim_per_head) -> None:
        super(TransLayer, self).__init__()
        D = n_head * n_dim_per_head

        self.num_node = num_node
        self.n_predict = n_predict
        self.n_history = n_history
        self.n_head = n_head
        self.n_dim_per_head = n_dim_per_head

        self.input_linear = nn.Linear(1, n_predict)

        self.conv_q = nn.Conv2d(in_features, D, kernel_size=(1, 1))
        self.conv_k = nn.Conv2d(in_features, D, kernel_size=(1, 1))
        self.conv_v = nn.Conv2d(in_features, D, kernel_size=(1, 1))

        self.length_liner = nn.Linear(n_history, n_predict)
        self.linear = nn.Linear(D, n_dim_per_head)

        self.LayerNorm = nn.LayerNorm([n_dim_per_head, n_predict, num_node])

    def forward(self, inputs, find_seq):
        """
        inputs: (batch_size, 1, num_node, in_features)
        find_seq: (batch_size, n_history, num_node, in_features)

        """
        batch_size = inputs.shape[0]

        # (batch_size, in_features, num_node, 1)
        inputs = inputs.permute(0, 3, 2, 1)
        out = self.input_linear(inputs)
        # (batch_size, in_featrues, num_node, n_predict)

        ResConn = out[:]

        # (batch_size, in_features, num_node, num_seq)
        find_seq = find_seq.permute(0, 3, 2, 1)

        # (batch_size, n_head * n_dim_per_head, num_node, n_history)
        query = self.conv_q(find_seq)
        # (batch_size, n_head * n_dim_per_head, num_node, n_predict)
        key = self.conv_k(out)
        # (batch_size, n_head * n_dim_per_head, num_node, n_predict)
        value = self.conv_v(out)

        # (n_head * batch_size, n_dim_per_head, num_node, timestamp)
        query = torch.concat(torch.split(
            query, self.n_dim_per_head, dim=1), dim=0)
        key = torch.concat(torch.split(key, self.n_dim_per_head, dim=1), dim=0)
        value = torch.concat(torch.split(
            value, self.n_dim_per_head, dim=1), dim=0)

        # (n_head * batch_size, num_node, n_history, n_dim_per_head)
        query = query.permute(0, 2, 3, 1)

        # (n_head * batch_size, num_node, n_dim_per_head, n_predict)
        key = key.permute(0, 2, 1, 3)

        # (n_head * batch_size, num_node, n_predict, n_dim_per_head)
        value = value.permute(0, 2, 3, 1)

        attention = torch.matmul(query, key)
        attention = attention / self.n_dim_per_head ** 0.5
        attention = torch.softmax(attention, dim=-1)

        result = torch.matmul(attention, value)

        result = torch.concat(torch.split(result, batch_size, dim=0), dim=-1)

        result = torch.relu(self.linear(result))
        # (batch_size, num_node, n_predict, n_dim_per_head)

        if self.n_history != self.n_predict:
            result = result.permute(0, 1, 3, 2)
            result = self.length_liner(result)
            result = result.permute(0, 1, 3, 2)

        result = result.permute(0, 3, 1, 2)
        # (batch_size, n_dim_per_head, num_node, n_predict)
        result = result + ResConn
        result = result.permute(0, 1, 3, 2)

        result = self.LayerNorm(result)
        # (batch_size, n_dim_per_head, n_predict, num_node)

        return result.permute(0, 2, 3, 1)


class Archer(nn.Module):
    def __init__(self, num_node, num_embed, n_history, n_predict, in_features, mid_features, out_features, adj) -> None:
        super(Archer, self).__init__()
        self.pre_linear = nn.Linear(in_features, mid_features)

        self.adj = nn.Parameter(torch.from_numpy(adj), requires_grad=False)
        self.embed = nn.Parameter(torch.FloatTensor(num_node, num_embed))

        self.encoder = Encoder(n_history, n_predict,
                               mid_features, mid_features)

        self.trans = TransLayer(num_node, n_history=n_history,
                                n_predict=n_predict, in_features=mid_features, n_head=4, n_dim_per_head=mid_features)

        self.gcn = GCNCell(mid_features, mid_features)

        self.gcn_with_embed = GCNCell(mid_features, mid_features)
        self.leakyRelu = nn.ReLU()

        self.downsample = nn.Conv2d(
            2 * mid_features, mid_features, kernel_size=1)

        self.fully = nn.Linear(mid_features, mid_features)
        self.predict = nn.Linear(mid_features, out_features)

        self.reset_parameter()

    def reset_parameter(self):
        self.embed.data.normal_()

    def forward(self, inputs):

        out = torch.relu(self.pre_linear(inputs))
        inputs = out

        out = self.encoder(out)
        print('tcn:',out.shape)

        out = self.trans(out, inputs)
        print('trans:', out.shape)

        ResConn = out[:]

        dynamic_adj = torch.softmax(self.leakyRelu(torch.matmul(
            self.embed, self.embed.T)), dim=1)
        embed_out = self.gcn_with_embed(out, dynamic_adj)

        out = self.gcn(out, self.adj)

        out = torch.cat([embed_out, out], dim=-1)

        out = out.permute(0, 3, 2, 1)
        out = self.downsample(out)
        out = out.permute(0, 3, 2, 1)

        out = out + ResConn

        out = torch.relu(self.fully(out))
        out = self.predict(out)
        return out


# gen adj
    def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance") -> np.array:
        """
        :param distance_file: str, path of csv file to save the distances between nodes.
        :param num_nodes: int, number of nodes in the graph
        :param id_file: str, path of txt file to save the order of the nodes.
        :param graph_type: str, ["connect", "distance"]
        :return:
            np.array(N, N)
        """
        A = np.zeros([int(num_nodes), int(num_nodes)])

        if id_file:
            with open(id_file, "r") as f_id:

                node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

                with open(distance_file, "r") as f_d:
                    f_d.readline()
                    reader = csv.reader(f_d)
                    for item in reader:
                        if len(item) != 3:
                            continue
                        i, j, distance = int(item[0]), int(item[1]), float(item[2])
                        if graph_type == "connect":
                            A[node_id_dict[i], node_id_dict[j]] = 1.
                            A[node_id_dict[j], node_id_dict[i]] = 1.
                        elif graph_type == "distance":
                            A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                            A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                        else:
                            raise ValueError("graph type is not correct (connect or distance)")
            return A

        with open(distance_file, "r") as f_d:
            f_d.readline()
            reader = csv.reader(f_d)
            for item in reader:
                if len(item) != 3:
                    continue
                i, j, distance = int(item[0]), int(item[1]), float(item[2])

                if graph_type == "connect":
                    A[i, j], A[j, i] = 1., 1.
                elif graph_type == "distance":
                    A[i, j] = 1. / distance
                    A[j, i] = 1. / distance
                else:
                    raise ValueError("graph type is not correct (connect or distance)")

        return torch.from_numpy(A)

if __name__ == '__main__':
    distance_file = '../data/PEMS03/distance.csv'
    print(distance_file)
    import csv

    A = np.zeros([int(358), int(358)])

    with open('../data/PEMS03/PEMS03.txt', 'r') as f:
        id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

    with open(distance_file, 'r') as f:
        f.readline()  # 略过表头那一行
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            # A[id_dict[i], id_dict[j]] = 1
            A[id_dict[i], id_dict[j]] = distance
            if distance<=0.:
                print(distance)
                print(i,j)
                print("aaa")

    print(A.shape)

    exit()


    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        max1=0
        min1=100000000
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])
            if i>max1:
                max1=i
            if min1>i:
                min1=i
            if j>max1:
                max1=j
            if min1>j:
                min1=j
        print(max1,min1)
        print(max1-min1)

            # if graph_type == "connect":
            #     A[i, j], A[j, i] = 1., 1.
            # elif graph_type == "distance":
            #     A[i, j] = 1. / distance
            #     A[j, i] = 1. / distance
            # else:
            #     raise ValueError("graph type is not correct (connect or distance)")



    exit()

    adj = np.random.randn(307, 307).astype(np.float32)
    data = torch.randn(64, 12, 307, 1)
    net = Archer(307,2,12, 12, 1, 32, 1, adj)
    tcn=Encoder(12,12,1,32)
    out=tcn(data)
    print(out.shape)
    total = 0
    # for param in net.parameters():
    #     total += param.numel()
    # print(total)

# if __name__=='__main__':
#     tcn=Encoder(12,12,1,32)
#     data=torch.randn((64,12,170,1))
#     print(data.shape)
#     out=tcn(data)
#     print(out.shape)