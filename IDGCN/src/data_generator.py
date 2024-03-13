from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import random
from tqdm import tqdm
import scipy
import torch
from scipy.stats import entropy

def get_user2items_dict(data_dict):
    max_user = 0
    max_item = 0
    user2items_dict = defaultdict(list)
    for u_id, i_id in zip(data_dict["user_id"],data_dict["item_id"]):
        user2items_dict[u_id].append(i_id)
        if u_id>max_user:
            max_user = u_id
        if i_id>max_item:
            max_item = i_id
    print("max_user: ", max_user)
    print("max_item: ", max_item)
    return user2items_dict

def get_user2items_group_dict(data_dict, num_group=10):
    user2items_group_dict = [defaultdict(list) for i in range(num_group)]
    for u_id, i_id, g_id in zip(data_dict["user_id"],data_dict["item_id"],data_dict["group_id"]):
        user2items_group_dict[int(g_id)][u_id].append(i_id)
    return user2items_group_dict

def get_item2users_dict(data_dict):
    item2users_dict = defaultdict(list)
    for u_id, i_id in zip(data_dict["user_id"],data_dict["item_id"]):
        item2users_dict[i_id].append(u_id)
    return item2users_dict

def load_data(data_path, sep=",", usecols=None):
    logging.info("Reading file: " + data_path)
    ddf = pd.read_csv(data_path, sep=sep, usecols=usecols)
    data_dict = dict()
    for feature in usecols:
        data_dict[feature] = ddf.loc[:, feature].values
    num_samples = len(list(data_dict.values())[0])
    return data_dict, num_samples



class TrainDataset(Dataset):
    def __init__(self,data_path):
        self.data_dict, self.num_samples = load_data(data_path, usecols=["user_id","item_id","label"])
        self.labels = self.data_dict["label"]
        self.users = self.data_dict["user_id"]
        self.items = self.data_dict["item_id"]
        self.pos_users = self.data_dict["user_id"]
        self.pos_items = self.data_dict["item_id"]
        self.neg_items = None
        self.all_items = None
        self.num_items = len(set(self.data_dict["item_id"]))
        self.num_users = len(set(self.data_dict["user_id"]))
        self.max_user = np.max(self.users)
        self.max_item = np.max(self.items)


    def __getitem__(self, index):
        return self.pos_users[index], self.pos_items[index], self.neg_items[index]

    def __len__(self):
        return self.num_samples

class TrainGenerator(DataLoader):
    def __init__(self, model_name, data_path, batch_size=32, shuffle=True,   ###!!!!修改，之前是True
                 user_sample_rate=None, item_sample_rate=None, item_freq=None, user_freq=None, category_dict=None, new_item2cate=[],
                 num_neg=None,num_pos=None, num_pos_user=None, tau=None, mode="diverse",**kwargs):
        self.model_name = model_name
        self.dataset = TrainDataset(data_path)
        super(TrainGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                             shuffle=shuffle)
        self.set_item_ids = set(self.dataset.items)
        self.user2items_dict = get_user2items_dict(self.dataset.data_dict)
        self.item2users_dict = get_item2users_dict(self.dataset.data_dict)

        print('all items in training set:', len(self.item2users_dict))



        self.num_samples = self.dataset.num_samples
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / batch_size))

        self.num_items = self.dataset.num_items
        self.num_users = self.dataset.num_users
        self.num_nodes = self.num_users + self.num_items

        self.max_user = self.dataset.max_user
        self.max_item = self.dataset.max_item
        self.item_sample_rate = item_sample_rate
        self.user_sample_rate = user_sample_rate
        self.item_freq = item_freq
        self.user_freq = user_freq

        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_pos_user = num_pos_user
        self.tau = tau
        self.mode = mode

        print("self.pos_num: ",self.num_pos)
        print("self.num_negs: ", self.num_neg)
        self.category_dict = category_dict  #item category dict
        self.new_item2cate = new_item2cate

        if self.item_sample_rate:
            self.get_user2items_sample_rate_dict()
        if self.model_name=="lightgcn" or  self.model_name=="algcn"\
                or self.model_name=='idgcn':
            self.adj_mat = self.generate_ori_norm_adj()
        else:
            self.adj_mat = self.build_power_sparse_graph()  # adj_mat


    def generate_ori_norm_adj(self):
        user_nodes, item_nodes = self.to_node(self.user2items_dict)
        edge_index, edge_weight = self.to_edge(user_nodes, item_nodes)
        norm_adj = self.generate(edge_index, edge_weight)
        return norm_adj

    def generate(self, edge_index, edge_weight):
        # edge_index, edge_weight = self.add_self_loop(edge_index, edge_weight)  #是否添加自环?
        edge_index, edge_weight = self.norm(edge_index, edge_weight)
        return self.mat(edge_index, edge_weight)

    def to_node(self, train_u2i):
        node1, node2 = [], []
        for i, j in train_u2i.items():
            node1.extend([i] * len(j))
            node2.extend(j)

        node1 = torch.tensor(node1, dtype=torch.long)
        node2 = torch.tensor(node2, dtype=torch.long)
        return node1, node2

    def mat(self, edge_index, edge_weight):
        return torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([self.num_nodes, self.num_nodes]))
    def to_edge(self, train_u, train_i):
        row = torch.cat([train_u, train_i + self.num_users])
        col = torch.cat([train_i + self.num_users, train_u])
        edge_index = torch.stack([row, col]).to(torch.long)
        edge_weight = torch.ones_like(row).to(torch.float32)
        return edge_index, edge_weight


    def add_self_loop(self, edge_index, edge_weight):
        """ add self-loop """

        loop_index = torch.arange(0, self.num_nodes, dtype=torch.long)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(self.num_nodes, dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_index], dim=-1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=-1)

        return edge_index, edge_weight

    def norm(self, edge_index, edge_weight):
        """ D^{-1/2} * A * D^{-1/2}  """
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(self.num_nodes, dtype=torch.float32)
        deg = deg.scatter_add(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    def get_user_diverse(self):
        user_diverse_weight = dict(zip(range(0, self.num_users), [0] * self.num_users))
        for u, item_set in self.user2items_dict.items():
            unique_item_cate = set()
            for item in item_set:
                unique_item_cate.add(self.category_dict[item])
            if len(item_set) > 0:
                user_diverse_weight[u] = len(unique_item_cate) / len(item_set)
            else:
                user_diverse_weight[u] = 0
        d_weight = list(user_diverse_weight.values())

        # ####new added normalized diverse weight
        # max_ = max(d_weight)
        # min_ = min(d_weight)
        # normalized_d_weight = [(x-min_)/(max_-min_) for x in d_weight]
        # return torch.tensor(normalized_d_weight)
        return torch.tensor(d_weight)

    def get_user_entropy(self):
        user_entropy_weight = dict(zip(range(0, self.num_users), [0] * self.num_users))
        for u, item_set in self.user2items_dict.items():
            stat = [self.category_dict[item] for item in item_set]
            stat = np.unique(np.array(stat), return_counts=True)[1]
            entropy_u = entropy(stat)
            user_entropy_weight[u] = entropy_u
        # d_weight = list(user_diverse_weight.values())
        # max_ = max(d_weight)
        # min_ = min(d_weight)
        # normalized_d_weight = [(x - min_) / (max_ - min_) for x in d_weight]
        return torch.tensor(list(user_entropy_weight.values()))



    def get_user2items_sample_rate_dict(self):
        self.user2items_sam_rate_dict = defaultdict(list)
        for u in self.user2items_dict.keys():
            items = self.user2items_dict[u]
            for i in items:
                self.user2items_sam_rate_dict[u].append(self.item_sample_rate[i])
            if len(items)>0:
                S = sum(self.user2items_sam_rate_dict[u])
                self.user2items_sam_rate_dict[u] = [i/S for i in self.user2items_sam_rate_dict[u]]


    def __iter__(self):  #iteration step
        if self.num_pos>1:
            self.rns_item_sampling()
        if self.num_pos_user>1:
            self.rns_user_sampling()
        ##if it was num_pos=1, num_pos_user=1, then regular negative sampling
        self.negative_sampling()
        iter = super(TrainGenerator, self).__iter__()
        while True:
            try:
                yield next(iter) # a batch iterator
            except StopIteration:
                break

    def __len__(self):
        return self.num_batches

    def negative_sampling(self):
        #### neg_item = np.empty((self.num_samples, 1), dtype=np.int)
        ###### =======original=========
        if self.num_neg > 0:
            neg_item_indexes = np.random.choice(self.num_items,
                                                size=(self.num_samples, self.num_neg),
                                                replace=True)
            self.dataset.neg_items = neg_item_indexes   #original
        #     neg_item = np.hstack((neg_item_indexes, neg_item))
        #
        #     #####self.dataset.all_items = np.hstack([self.dataset.items.reshape(-1, 1),neg_item_indexes])
        # if self.mode=="diverse":
        #     for i in range(0,4):
        #         sampled_item_index = np.random.choice(self.new_item2cate[i], size=(self.num_samples, 2), replace=True)
        #         neg_item = np.hstack((sampled_item_index, neg_item))
        #     neg_item = np.delete(neg_item, -1, axis=1)
        # else:
        #     neg_item = np.delete(neg_item, -1, axis=1)
        # self.dataset.neg_items = neg_item
        tqdm.write("negative sampling.")


    def uniform_resampling(self,):
        if self.num_pos!=-1:
            # sample users:
            train_users_set = list(set(self.dataset.data_dict["user_id"]))
            if self.user_sample_rate:
                users_index = np.random.choice(len(train_users_set), self.dataset.num_samples, p=self.user_sample_rate)
            else:
                users_index = np.random.choice(len(train_users_set), self.dataset.num_samples)
            self.dataset.pos_users = np.array(train_users_set)[users_index]

            # sample items:
            items = []
            if self.item_sample_rate:
                for k,u in enumerate(self.dataset.pos_users):
                    pos_items = self.user2items_dict[u]
                    pos_items_rate = self.user2items_sam_rate_dict[u]
                    items.append(np.random.choice(pos_items,size=(self.num_pos),p=pos_items_rate))
            else:
                for k,u in enumerate(self.dataset.pos_users):
                    pos_items = self.user2items_dict[u]
                    items.append(np.random.choice(pos_items,size=(self.num_pos)))
            self.dataset.pos_items = np.array(items)

        if self.num_pos_user>1:
            users = []
            if self.num_pos < 2:
                for u,i in zip(self.dataset.pos_users, self.dataset.pos_items):
                    pos_users = self.item2users_dict[int(i)]
                    users.append([u, np.random.choice(pos_users).tolist()])
            else:
                for u,i in zip(self.dataset.pos_users, self.dataset.pos_items):
                    pos_users = self.item2users_dict[i[0]]
                    users.append([u, int(np.random.choice(pos_users))])
            self.dataset.pos_users = np.array(users)

    def rns_item_sampling(self):
        items = []
        for u,i in zip(self.dataset.users, self.dataset.items):
            pos_items = self.user2items_dict[u]
            #items.append([i, int(np.random.choice(pos_items))])
            items.append(np.append([i], np.random.choice(pos_items, self.num_pos-1)))
        self.dataset.pos_items = np.array(items)
        tqdm.write("rns item sampling.")

    def rns_user_sampling(self):
        users = []
        for u,i in zip(self.dataset.users, self.dataset.items):
            pos_users = self.item2users_dict[i]
            #users.append([u, int(np.random.choice(pos_users))])
            users.append(np.append([u], np.random.choice(pos_users, self.num_pos_user-1)))
        tqdm.write("rns user sampling.")
        self.dataset.pos_users = np.array(users)


    def update(self, delta = 1):
        if self.user_freq:
            print("delta: ", delta)
            def f(x, delta):
                if x > 0:
                    x = x ** delta
                return x

            self.user_sample_rate = [f(i,delta) for i in self.user_freq]
            sum_freq = sum(self.user_sample_rate)
            self.user_sample_rate = [i/sum_freq for i in self.user_sample_rate]

    def build_power_sparse_graph(self):
        a = np.array([self.dataset.users])   #(1, 62491)
        b = np.array([self.dataset.items])   #(1, 62491)

        data_cf = np.append(a.T, b.T, axis=1)
        n_users = self.max_user + 1
        n_items = self.max_item + 1

        def _bi_norm_laplacian(adj):
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -1).flatten()     #D^(-1)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.          #无穷大的值赋值为0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)      #对角矩阵D
            tau = self.tau
            print("tau:", tau)
            d_inv_log = np.power(rowsum, tau).flatten()  #power influence  F(D)
            d_inv_log = sp.diags(d_inv_log)    #另一个对角矩阵


            bi_lap = d_mat_inv_sqrt.dot(adj)   #D^(-1) * A
            bi_lap = d_inv_log.dot(bi_lap)
            #bi_lap = bi_lap.dot(d_inv_log)    #F(D)*D^(-1)*A
            return bi_lap.tocoo()
        cf = data_cf.copy()  #(62491, 2)

        cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
        cf_ = cf.copy()
        cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

        cf_ = np.concatenate([cf, cf_], axis=0)  # A equation 1 in papers [[0, R], [R^T, 0]]

        # print(cf_.shape)   #(124982, 2)
        vals = [1.] * len(cf_)
        mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users + n_items, n_users + n_items))
        logging.info("create adj_mat successfully.")
        return _bi_norm_laplacian(mat)


    def build_sparse_graph(self):
        a = np.array([self.dataset.users])   #(1, 62491)
        b = np.array([self.dataset.items])   #(1, 62491)
        data_cf = np.append(a.T, b.T, axis=1)
        # n_users = self.max_user + 1
        # n_items = self.max_item + 1
        def _bi_norm_laplacian(adj):
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()     #D^(-1/2)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.          #无穷大的值赋值为0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)      #对角矩阵D

            bi_lap = d_mat_inv_sqrt.dot(adj)   # D^(-1/2) * A
            bi_lap = bi_lap.dot(d_mat_inv_sqrt)  # D^(-1/2)*A * D^(-1/2)
            return bi_lap.tocoo()
        cf = data_cf.copy()  #(62491, 2)
        cf[:, 1] = cf[:, 1] + self.n_users  # [0, n_items) -> [n_users, n_users+n_items)
        cf_ = cf.copy()
        cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

        cf_ = np.concatenate([cf, cf_], axis=0)  # A equation 1 in papers [[0, R], [R^T, 0]]

        # print(cf_.shape)   #(124982, 2)
        vals = [1.] * len(cf_)
        mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        logging.info("create adj_mat successfully.")
        return _bi_norm_laplacian(mat)




class TestDataset(Dataset):
    def __init__(self, data_path, col_name, usecols):
        self.col_name = col_name
        self.data_dict, self.num_samples = load_data(data_path, usecols=usecols)

    def __getitem__(self, index):
        return self.data_dict[self.col_name][index]

    def __len__(self):
        return self.num_samples




class TestGenerator(object):
    def __init__(self, data_path, item_corpus_path, batch_size=256, shuffle=False):
        user_dataset = TestDataset(data_path, col_name="user_id", usecols=["user_id","item_id","label"])

        self.user2items_dict = get_user2items_dict(user_dataset.data_dict)

        # group test dataset
        #user_group_dataset = TestDataset(data_path, col_name="user_id", usecols=["user_id", "item_id", "label", "group_id"])
        #self.user2items_group_dict = get_user2items_group_dict(user_group_dataset.data_dict)
        self.user2items_group_dict = None

        # pick users of unique query_index
        self.test_users, _ = np.unique(user_dataset.data_dict["user_id"],
                                                    return_index=True)
        user_dataset.num_samples = len(self.test_users)
        self.num_samples = len(user_dataset)
        user_dataset.data_dict["user_id"] = self.test_users

        item_dataset = TestDataset(item_corpus_path, "item_id", usecols=["item_id"])
        self.user_loader = DataLoader(dataset=user_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=0)
        self.item_loader = DataLoader(dataset=item_dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=0)

def data_generator(model_name, train_data_path, valid_data_path, item_corpus_path, test_data_path, batch_size=256, num_negs=None,item_cate_dict=dict(),new_item2cate=[], mode="", **kwargs):
    # test_gen = None
    # print("in data_generator: lenght of item_Cate_dict: ", len(item_cate_dict))
    # batch_size = 16
    train_gen = TrainGenerator(model_name, data_path=train_data_path, batch_size= batch_size, num_negs=num_negs, category_dict=item_cate_dict,new_item2cate=new_item2cate,mode=mode, **kwargs)
    valid_gen = TestGenerator(data_path=valid_data_path, item_corpus_path=item_corpus_path, batch_size= batch_size)
    test_gen = TestGenerator(data_path = test_data_path, item_corpus_path = item_corpus_path, batch_size = batch_size)

    return train_gen, valid_gen, test_gen

# def train_data_generator(train_data_path, item_sample_rate=None, batch_size=None, num_negs=None, item_cate_dict=dict(), **kwargs):
#     print("num_negs: ", num_negs)
#     train_gen = TrainGenerator(data_path=train_data_path, batch_size= batch_size, item_sample_rate=item_sample_rate, num_negs=num_negs, category_dict=item_cate_dict, **kwargs)
#
#     return train_gen
