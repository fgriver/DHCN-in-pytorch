import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


class HyperGrapyConv(nn.Module):
    def __init__(self, hg_conv_num_layer):
        super(HyperGrapyConv, self).__init__()
        self.layer_nums = hg_conv_num_layer

    def forward(self, hg_adj_matrix, items_emb):
        item_embedding_list = [items_emb]
        for i in range(self.layer_nums):
            new_items_emb = torch.sparse.mm(hg_adj_matrix, items_emb)
            item_embedding_list.append(new_items_emb)
        item_embedding_list = torch.stack(item_embedding_list)
        final_sess_emb = torch.mean(item_embedding_list, dim=0)
        return final_sess_emb


class LineGraphConv(nn.Module):
    def __init__(self, line_conv_num_layers):
        super(LineGraphConv, self).__init__()
        self.layer_nums = line_conv_num_layers

    def forward(self, D, A, items_embedding, padded_session_info, sessions_len):
        """
        :param D: 度矩阵
        :param A: 邻接矩阵
        :param items_embedding: 项目集的嵌入
        :param padded_session_info: 会话信息
        :param sessions_len: 序列实际长度
        :return:
        """
        DA = torch.mm(D, A).float()
        zeros = torch.zeros((1, items_embedding.shape[1]), dtype=torch.float32, device='cuda')
        items_embedding = torch.cat([zeros, items_embedding], 0)
        # 按照 session 中 items 的顺序, 保存相对应的item_embs
        sess_info_emb = []
        for i in range(padded_session_info.shape[0]):
            sess_info_emb.append(torch.index_select(items_embedding, 0, padded_session_info[i].long()))

        sess_line_conv_emb = torch.stack(sess_info_emb, 0)
        sess_line_conv_emb = torch.div(torch.sum(sess_line_conv_emb, 1), sessions_len.view(-1, 1))

        conv_embs = [sess_line_conv_emb]

        for i in range(self.layer_nums):
            old_conv_emb = conv_embs[-1]
            new_conv_emb = torch.mm(DA, old_conv_emb)
            conv_embs.append(new_conv_emb)

        sess_final_emb = torch.stack(conv_embs)
        sess_final_emb = torch.mean(sess_final_emb, dim=0)

        return sess_final_emb


class DHCN(nn.Module):
    def __init__(self, opt, n_node, hg_adj_matrix):
        super(DHCN, self).__init__()
        self.batch_size = opt.batch_size
        self.hidden_size = opt.hidden_size

        values = hg_adj_matrix.data
        indices = np.vstack((hg_adj_matrix.row, hg_adj_matrix.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = hg_adj_matrix.shape
        adjacency = torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.float32, device='cuda')
        # adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.hg_adj_matrix = adjacency

        self.n_node = n_node
        self.items_embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.position_embedding = nn.Embedding(200, self.hidden_size)

        self.hgcn = HyperGrapyConv(opt.hg_conv_step)
        self.w1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.w3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_T = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.lgcn = LineGraphConv(opt.line_conv_steps)
        self.beta = opt.beta
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_hg_emb(self, hg_conved_items_embeddig, sessions_len, sess_info, reversed_sess_info, masks):
        """
        :param hg_conved_items_embeddig: 经过超图卷积过后的项目嵌入 Tensor
        :param sessions_len: 每个会话对应的实际长度， 用于计算 平均会话嵌入
        :param sess_info: 会话信息（已经经过填充的）
        :param reversed_sess_info: 反向的会话信息（已经经过填充的）
        :param masks: （遮掩项，用于屏蔽 pos_emb 对应的 空白填充项）
        :return:
        """
        zeros = torch.zeros((1, self.hidden_size), dtype=torch.float32, device='cuda')
        hg_conved_items_embeddig = torch.cat([zeros, hg_conved_items_embeddig], 0)

        seq_h = []
        get = lambda i: hg_conved_items_embeddig[reversed_sess_info[i].long()]
        for i in torch.arange(sess_info.shape[0]):
            seq_h.append(get(i))
        # x_t
        seq_h = torch.stack(seq_h)
        length = seq_h.shape[1]
        sess_mean_hidden = torch.div(torch.sum(seq_h, 1), sessions_len.view(-1, 1))  # b x h
        sess_mean_hidden = sess_mean_hidden.unsqueeze(1).repeat(1, length, 1)  # b x len x h

        pos_emb = self.position_embedding.weight[:length]
        pos_emb = pos_emb.unsqueeze(0).repeat(sess_mean_hidden.shape[0], 1, 1)
        # x_t^*
        # tag:
        new_seq_hidden = torch.tanh(self.w1(torch.cat([pos_emb, seq_h], -1)))

        temp_variable = torch.sigmoid(self.w2(sess_mean_hidden) + self.w3(new_seq_hidden))
        alpha = torch.matmul(temp_variable, self.f_T)
        masks = masks.float().unsqueeze(-1)
        alpha = alpha * masks

        theta = torch.sum((alpha * new_seq_hidden), 1)

        return theta

    def SSL(self, hg_conved_session_emb, line_conved_session_emb):
        def shuffle_row_column(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            test = torch.randperm(corrupted_embedding.size()[1])
            corrupted_embedding = corrupted_embedding[:, test]
            return corrupted_embedding

        def score(x, y):
            return torch.sum(torch.mul(x, y), 1)

        pos_emb = score(hg_conved_session_emb, line_conved_session_emb)
        neg_emb = score(shuffle_row_column(hg_conved_session_emb), line_conved_session_emb)

        ones = torch.ones(pos_emb.shape[0], dtype=torch.float32, device='cuda')
        # previous loss function error
        cl_loss = torch.sum(
            -torch.log(1e-8 + torch.sigmoid(pos_emb)) - torch.log(1e-8 + (ones - torch.sigmoid(neg_emb))))

        return cl_loss

    def forward(self, session_info, reversed_info, sessions_len, masks, line_adj_matrix, degree):
        """
        :param session_info: 'to generate hg_conved_session_emb'
        :param reversed_info: 'to generate hg_conved_session_emb'
        :param sessions_len: 'to generate hg_conved_session_emb'
        :param masks: 'to generate hg_conved_session_emb'

        :param line_adj_matrix: 'to generate line_conved_session_emb'
        :param degree: 'to generate line_conved_session_emb'
        :return:
            hg_conved_session_emb, cl_loss, hg_conved_item_emb
        """

        hg_conved_item_emb = self.hgcn(self.hg_adj_matrix, self.items_embedding.weight)

        hg_conved_session_emb = self.generate_hg_emb(hg_conved_item_emb,
                                                     sessions_len,
                                                     session_info,
                                                     reversed_info,
                                                     masks)

        line_conved_session_emb = self.lgcn(degree, line_adj_matrix, self.items_embedding.weight, session_info,
                                            sessions_len)

        cl_loss = self.SSL(hg_conved_session_emb, line_conved_session_emb)
        cl_loss = self.beta * cl_loss

        return hg_conved_session_emb, cl_loss, hg_conved_item_emb
