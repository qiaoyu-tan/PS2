import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
# from model_layer import GCNConv as GCNConv_layer
from torch.nn import Linear
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class GCN_tune(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 dropout, src_layer, dst_layer):
        super(GCN_tune, self).__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        num_layers = max(self.src_layer, self.dst_layer)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        return x_hidden[self.src_layer - 1], x_hidden[self.dst_layer - 1]


class SAGE_tune(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 dropout, src_layer, dst_layer):
        super(SAGE_tune, self).__init__()

        self.src_layer = src_layer
        self.dst_layer = dst_layer
        num_layers = max(self.src_layer, self.dst_layer)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        return x_hidden[self.src_layer - 1], x_hidden[self.dst_layer - 1]


class AutoLink_l2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type):
        super(AutoLink_l2, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.num_layers = 3
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def compute_loss(self, h, atten_matrix, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, atten_matrix, train_edge)
        neg_out = self.compute_pred(h, atten_matrix, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        # if torch.isnan(loss):
        #     print('----nan occur in pos loss={}'.format(pos_loss.item()))
        #     print('----nan occur in neg loss={}'.format(neg_loss.item()))
        #     print('### pos_out is \n')
        #     print(pos_out.data.cpu().numpy())
        #     print('### neg_out is \n')
        #     print(neg_out.data.cpu().numpy())
        #     pass
        return loss

    def compute_pred(self, h, atten_matrix, train_edge):
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h[train_edge[0]], h[train_edge[1]])
        return pos_pred

class SearchGraph_l2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_nodes,
                 temperature=0.07):
        super(SearchGraph_l2, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        self.arc = torch.nn.Parameter(torch.ones(size=[num_nodes, hidden_channels], dtype=torch.float) / self.num_nodes)
        # self.trans = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     if i == 0:
        #         self.trans.append(Linear(in_channels, hidden_channels, bias=False))
        #     else:
        #         self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        # self.trans.append(Linear(hidden_channels, 1, bias=False))

    # def reset_parameters(self):
    #     for conv in self.trans:
    #         conv.reset_parameters()

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        # for conv in self.trans[:-1]:
        #     x = conv(x)
        #     x = F.relu(x)
        # x = self.trans[-1](x)
        # x = torch.squeeze(self.arc, dim=2)
        x = self.arc
        arch_set = torch.softmax(x / self.temperature, dim=1)
        device = arch_set.device
        if not self.training:
            n, c = arch_set.shape
            eyes_atten = torch.eye(c)
            atten_, atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
            arch_set = arch_set.to(device)

        return arch_set


class SearchGraph_rs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, arch_layers,
                 temperature=0.07):
        super(SearchGraph_rs, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.arch_layers = arch_layers

        self.search_len = num_layers**2
        self.search_space = torch.eye(self.search_len)
        self.choice_list = torch.arange(self.search_len, dtype=torch.float)

    def forward(self, x, grad=False):
        # x with shape [batch, num_layer, dim]
        n, c, d = x.shape
        device = x.device
        # rs_indice = torch.multinomial(self.choice_list, n, replacement=True)
        rs_indice = torch.randint(0, self.search_len, (n,), dtype=torch.long)
        arch_set = self.search_space[rs_indice]
        arch_set = arch_set.to(device)
        return arch_set


class SearchGraph_qa(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, arch_layers,
                 temperature=0.07):
        super(SearchGraph_qa, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.arch_layers = arch_layers

        self.search_len = sum([num_layers - i for i in range(num_layers)])
        self.search_space = torch.eye(self.search_len)
        self.choice_list = torch.arange(self.search_len, dtype=torch.float)

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        n, c, d = x.shape
        device = x.device
        # rs_indice = torch.multinomial(self.choice_list, n, replacement=True)
        rs_indice = torch.randint(0, self.search_len, (n,), dtype=torch.long)
        arch_set = self.search_space[rs_indice]
        arch_set = arch_set.to(device)
        return arch_set


class SearchGraph_l22(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 temperature=0.07):
        super(SearchGraph_l22, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers

        self.trans = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for conv in self.trans:
            conv.reset_parameters()

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        for conv in self.trans[:-1]:
            x = conv(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x, dim=2)
        arch_set = torch.softmax(x / self.temperature, dim=1)
        device = arch_set.device
        if not self.training:
            n, c = arch_set.shape
            eyes_atten = torch.eye(c)
            atten_, atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
            arch_set = arch_set.to(device)

        return arch_set


class AutoLink_l3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_l3, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, True)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_discrete(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, False)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, False)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3Seal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_l3Seal, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels
        arch_vector = []
        for i in range(1, num_layers + 1):
            for j in range(1, num_layers + 1):
                arch_vector.append([i, j])
        self.arch_vector = torch.from_numpy(np.array(arch_vector))

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, True)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_edge(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        _, index_sub = torch.max(atten_matrix, dim=1)
        subgraphs = self.arch_vector[index_sub]
        return subgraphs

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3scale(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_l3scale, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv_layer(in_channels, hidden_channels))
                else:
                    self.convs.append(GCNConv_layer(hidden_channels, hidden_channels))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def complte_forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def compute_loss(self, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(arch_net, train_edge, True)
        neg_out = self.compute_pred(arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(pos_atten, train_edge)
        neg_out = self.compute_pred_arch(neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, arch_net, train_edge, grad=False):
        h = self.cross_pair(train_edge[0], train_edge[1])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, atten_matrix, train_edge):
        h = self.cross_pair(train_edge[0], train_edge[1])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, arch_net, train_edge):
        h = self.cross_pair(train_edge[0], train_edge[1])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3Table(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, num_node, lin_layers=2, cat_type='multi'):
        super(AutoLink_l3Table, self).__init__()
        self.gnn_type = gnn_type
        self.num_node = num_node
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.hidden_channels = hidden_channels
        self.cat_type = cat_type

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.x = torch.nn.Embedding(num_node, in_channels)
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.x.weight)

    def forward(self, adj_t):
        x = self.x.weight
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_discrete(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, False)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, False)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3TableSeal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, num_node, lin_layers=2, cat_type='multi'):
        super(AutoLink_l3TableSeal, self).__init__()
        self.gnn_type = gnn_type
        self.num_node = num_node
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.hidden_channels = hidden_channels
        self.cat_type = cat_type
        arch_vector = []
        for i in range(1, num_layers + 1):
            for j in range(1, num_layers + 1):
                arch_vector.append([i, j])
        self.arch_vector = torch.from_numpy(np.array(arch_vector))

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.x = torch.nn.Embedding(num_node, in_channels)
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.x.weight)

    def forward(self, adj_t):
        x = self.x.weight
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_edge(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        _, index_sub = torch.max(atten_matrix, dim=1)
        subgraphs = self.arch_vector[index_sub]
        return subgraphs

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg


class AutoLink_l3Rs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type):
        super(AutoLink_l3Rs, self).__init__()

        self.gnn_type = gnn_type
        self.num_layers = 2
        self.input_layer = num_layers
        self.hidden_channels = hidden_channels

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.x.weight)

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                x.append(x_i[:, i, :] * x_j[:, j, :])
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_pred(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred


class SearchGraph_l31(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, cat_type='multi',
                 temperature=0.07):
        super(SearchGraph_l31, self).__init__()
        self.temperature = temperature
        self.num_layers = num_layers
        self.cat_type = cat_type
        if self.cat_type == 'multi':
            in_channels = in_channels
        else:
            in_channels = in_channels * 2
        self.trans = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for conv in self.trans:
            conv.reset_parameters()

    def forward(self, x, grad=False):
        # x with shape [batch, num_layer, dim]
        for conv in self.trans[:-1]:
            x = conv(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x, dim=2)
        arch_set = torch.softmax(x / self.temperature, dim=1)
        if not self.training:
            if grad:
                return arch_set.detach()
            else:
                device = arch_set.device
                n, c = arch_set.shape
                eyes_atten = torch.eye(c)
                atten_, atten_indice = torch.max(arch_set, dim=1)
                arch_set = eyes_atten[atten_indice]
                arch_set = arch_set.to(device)
                return arch_set
        else:
            return arch_set


class AutoLink_Seal(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, gnn_type, lin_layers=3, cat_type='multi'):
        super(AutoLink_Seal, self).__init__()

        self.gnn_type = gnn_type
        # self.num_layers = num_layers
        self.lin_layers = lin_layers
        self.input_layer = num_layers
        self.cat_type = cat_type
        self.hidden_channels = hidden_channels
        arch_vector = []
        for i in range(1, num_layers + 1):
            for j in range(1, num_layers + 1):
                arch_vector.append([i, j])
        self.arch_vector = torch.from_numpy(np.array(arch_vector))

        self.convs = torch.nn.ModuleList()
        if self.gnn_type == 'SAGE':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(SAGEConv(in_channels, hidden_channels))
                else:
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
                else:
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        else:
            raise SystemExit('The gnn type should be GCN or SAGE')

        self._init_predictor()
        self.dropout = dropout

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.cat_type == 'multi':
            input_channels = self.hidden_channels
        else:
            input_channels = self.hidden_channels * 2
        self.lins.append(torch.nn.Linear(input_channels, self.hidden_channels))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        x_hidden = torch.stack(x_hidden, dim=1)
        return x_hidden

    def pred_pair(self, x):
        # x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.input_layer):
            for j in range(self.input_layer):
                if self.cat_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                else:
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def compute_loss(self, h, arch_net, train_edge, train_edge_neg):
        pos_out = self.compute_pred(h, arch_net, train_edge, True)
        neg_out = self.compute_pred(h, arch_net, train_edge_neg, True)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss_arch(self, h, pos_atten, neg_atten, train_edge, train_edge_neg):
        pos_out = self.compute_pred_arch(h, pos_atten, train_edge)
        neg_out = self.compute_pred_arch(h, neg_atten, train_edge_neg)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss

    def compute_pred(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_edge(self, h, arch_net, train_edge, grad=False):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h, grad)
        _, index_sub = torch.max(atten_matrix, dim=1)
        subgraphs = self.arch_vector[index_sub]
        return subgraphs

    def compute_pred_arch(self, h, atten_matrix, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        h = torch.sum(h, dim=1)
        pos_pred = self.pred_pair(h)
        return pos_pred

    def compute_arch_input(self, h, arch_net, train_edge):
        h = self.cross_pair(h[train_edge[0]], h[train_edge[1]])
        atten_matrix = arch_net(h)
        return atten_matrix

    def compute_arch(self, h, arch_net, train_edge, train_edge_neg):
        atten_matrix_pos = self.compute_arch_input(h, arch_net, train_edge)
        atten_matrix_neg = self.compute_arch_input(h, arch_net, train_edge_neg)
        return atten_matrix_pos, atten_matrix_neg