import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, diags
from utils import compute_node_num

# 数据集中的 max_len 主要用于填充 items 和 session
def compute_max_len(rawdata):
    session_len = [len(session) for session in rawdata]
    max_len = max(session_len)
    return max_len


def compute_max_n_node(rawdata):
    session_n_node = [len(np.unique(session)) for session in rawdata]
    max_n_node = max(session_n_node)
    return max_n_node


def calculate_hg_adj_matrix(all_raw_data, batch_n_node):
    # 超图的邻接矩阵为全局性的
    """
    :param all_raw_data: batch中已经填充好的的session
    :param batch_n_node: batch中的最大节点数
    :return:
        hypergraph adj matrix
    """
    # indptr: csr_matrix中用于断点的下标;
    indptr, indices, weight = [0], [], []
    for session in all_raw_data:
        node = np.unique(session)
        length = len(node)
        s = indptr[-1]
        indptr.append((s + length))
        for j in range(length):
            # 添加的虽然是项目ID 但由于csr_matrix的下标模式 所以要将项目ID-1
            indices.append(node[j] - 1)
            weight.append(1)

    hg_adj_matrix = csr_matrix((weight, indices, indptr), shape=(len(all_raw_data), batch_n_node))
    return hg_adj_matrix


def calculate_line_adj_matrix(batch_padded_sess_info):
    """
    :param batch_padded_sess_info: batch中已经填充好的 items
    :return:
        line_adj_matrix: 超图的线性图邻接矩阵
        line_degree: 线性图的度矩阵
    """

    line_adj_matrix = np.zeros((len(batch_padded_sess_info), len(batch_padded_sess_info)))

    for i in range(len(batch_padded_sess_info)):
        session = set(batch_padded_sess_info[i])
        session.discard(0)
        for j in range(i + 1, len(batch_padded_sess_info)):
            next_session = set(batch_padded_sess_info[j])
            next_session.discard(0)
            sess_intersection = session.intersection(next_session)
            sess_union = session | next_session

            line_adj_matrix[i][j] = float(len(sess_intersection)) / float(len(sess_union))
            line_adj_matrix[j][i] = line_adj_matrix[i][j]

    line_adj_matrix = line_adj_matrix + np.diag([1.0] * len(batch_padded_sess_info))
    # 先求度
    degree = np.sum(np.array(line_adj_matrix), 1)
    # 再求度矩阵的逆矩阵
    line_degree = np.diag(1.0 / degree)

    return line_adj_matrix, line_degree


class MyDataset(Dataset):
    def __init__(self, rawdata, n_node):
        self.inputs = rawdata[0]

        self.n_node = n_node
        self.targets = rawdata[1]
        # 由于超图的复杂性 所以超图的邻接矩阵需要计算所有的数据
        H_T = calculate_hg_adj_matrix(self.inputs, self.n_node)

        epsilon = 1e-10

        D = np.sum(H_T.T, 1) + epsilon
        # todo: test_data -> D y
        D = diags(1.0 / D.A1)
        # print('******************')
        DH = D @ H_T.T
        # print('*-*-**-*-**-*-*-*-*')
        B = np.sum(H_T.T, 0) + epsilon
        B = diags(1.0 / B.A1)

        DHB = DH @ B
        DHBH = DHB @ H_T
        self.hg_adj_matrix = DHBH.tocoo()

    def __len__(self):
        return len(self.inputs)

    # 输出 items:会话中的项目 / mask: 原始会话中的掩码 / targets / A:邻接矩阵 / alias_inputs: 用于恢复原序列的别名列表
    # 需要为每个会画图构建图 并保存别名序列
    def __getitem__(self, index):
        """
        :param index: the index of batch; index = batch_size
        :return:
            item [ len]:
            mask [ len]:
            target [1]:
            alias_input [len]:
        """
        sess_len = len(self.inputs[index])
        session_info_input = self.inputs[index]
        target = self.targets[index] - 1

        return session_info_input, target, sess_len


def collate_fn(batch):
    """
    :param batch: -> inputs, masks, targets
    :return:
        targets:
        session_lens: to calculate the 'seq_mean_hidden'
        sessions_info: session info
        reversed_sessions_info: reversed session info
        masks:
        line_adj_matrix: via calculate_line_adj_matrix()
        degree: linear graph degree
    """
    session_info_inputs, targets, session_lens = zip(*batch)
    session_info_inputs, targets, session_lens = list(session_info_inputs), list(targets), list(session_lens)

    bacth_max_sess_len = max(session_lens)
    masks = []
    reversed_session_info = []

    for i, session in enumerate(session_info_inputs):
        session_info_inputs[i] = list(session) + (bacth_max_sess_len - session_lens[i]) * [0]
        masks.append(session_lens[i] * [1] + (bacth_max_sess_len - session_lens[i]) * [0])
        reversed_session_info.append(list(reversed(session)) + (bacth_max_sess_len - session_lens[i]) * [0])


    line_adj_matrix, degree = calculate_line_adj_matrix(session_info_inputs)

    return targets, session_lens, session_info_inputs, reversed_session_info, masks, line_adj_matrix, degree


def get_dataloader(dataset, batch_size):
    dataset2loader = MyDataset(rawdata=dataset)
    return DataLoader(dataset2loader, batch_size=batch_size, shuffle=True)


def split_validation(train_data, valid_rate):
    """
    :param train_data: tuple: train_data[0]->session; train_data[1]->target
    :param valid_rate: the rate of validation in all data set
    :return:
        (train_session_set, train_target_set), (valid_session_set, valid_target_set)
    """

    session_data, target_data = train_data
    num_samples = np.max(session_data)
    sidx = np.arange(len(session_data))
    np.random.shuffle(sidx)
    num_train = int(np.round(num_samples * (1. - valid_rate)))

    train_session_set = [session_data[s] for s in sidx[:num_train]]
    train_target_set = [target_data[s] for s in sidx[:num_train]]

    valid_session_set = [session_data[s] for s in sidx[num_train:]]
    valid_target_set = [target_data[s] for s in sidx[num_train:]]

    return (train_session_set, train_target_set), (valid_session_set, valid_target_set)
