'''
Created on October 1, 2020
@author: Tinglin Huang (huangtinglin@outlook.com)
'''
import random
import numpy as np
import torch
import torch.nn as nn
from src.models.base_model import BaseModel
import torch.nn.functional as F
from src.util.spmm import SpecialSpmm, CHUNK_SIZE_FOR_SPMM
import scipy.sparse as sp
import time
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
from torch import linalg as LA

class IDGCN(BaseModel):
    def __init__(self,
                 model_name=None,
                 mess_dropout_rate=0.1,
                 edge_dropout_rate = 0.5,
                 device=None,
                 adj_mat=None,
                 embedding_dim=64,
                 train_gen=None,
                 max_user_id=None,
                 max_item_id=None,
                 n_layers=0,
                 num_negs=0,
                 emb_lambda=0,
                 is_pos_item_mixing=False,
                 is_pos_user_mixing=False,
                 is_hard_negative_sampling=False,
                 is_hard_neg_mixing=False,
                 mixing_weight=None,
                 **param):
        super(IDGCN, self).__init__(device=device, **param)
        self.model_name = model_name
        self.n_users = max_user_id+1
        self.n_items = max_item_id+1

        self.mat = adj_mat.to(device)

        self.tttt = (self.mat.cpu().to_dense()[:self.n_users, self.n_users:self.n_users+self.n_items])  #(n_user, n_items)
        non_zero_indices = torch.nonzero(self.tttt).t()  #交互数
        values = self.tttt[tuple(non_zero_indices[i] for i in range(self.tttt.dim()))]
        self.sparse_interactions = torch.sparse_coo_tensor(non_zero_indices, values, self.tttt.size()).to(self.device)


        self.is_pos_item_mixing = is_pos_item_mixing
        self.is_pos_user_mixing = is_pos_user_mixing
        self.is_hard_negative_sampling = is_hard_negative_sampling
        self.is_hard_neg_mixing = is_hard_neg_mixing
        self.mixing_weight = mixing_weight
        self.emb_size = embedding_dim
        self.n_layers = n_layers
        print("n layer:", n_layers)
        self.mess_dropout = False
        self.mess_dropout_rate = mess_dropout_rate

        print("mess dropout_rate:", self.mess_dropout_rate)
        self.edge_dropout = False
        self.edge_dropout_rate = edge_dropout_rate
        self.n_negs = num_negs

        self.pretrain = False
        if self.pretrain:
            print("loading pretrained embeddings")
            path_u = "./data/lightgcn_AmazonBooks_user_embs_4.pt"
            path_i = "./data/lightgcn_AmazonBooks_item_embs_4.pt"
            user_emb = torch.load(path_u)
            item_emb = torch.load(path_i)
            self.user_embed = torch.nn.Embedding.from_pretrained(user_emb[:,0,:], freeze = True)
            self.item_embed = torch.nn.Embedding.from_pretrained(item_emb[:,0,:], freeze = True)
        else:
            self.user_embed = torch.nn.Embedding(self.n_users, self.emb_size)
            self.item_embed = torch.nn.Embedding(self.n_items, self.emb_size)
            self.user_weight = torch.nn.Embedding(self.n_users, 1)
            self.item_weight = torch.nn.Embedding(self.n_items, 1)
            self._init_weight()


        self.user_embedding = None
        self.item_embedding = None

        # self.gcn = self._init_model()
        self.to(device=self.device)

        self.my_dataloader()

        self.index, self.item__ = self.sparse_interactions.coalesce().indices()  # torch.Size([124680]) torch.Size([124680])
        self.index, self.item__ = self.index.tolist(), self.item__.tolist()


    def my_dataloader(self):
        # self.sparse_interactions = self._sparse_dropout(self.sparse_interactions, self.edge_dropout_rate) if edge_dropout else self.sparse_interactions
        # self.cf_dataloader = DataLoader(self.mat.coalesce().indices().t(), batch_size=10000, shuffle=False)
        self.cf_dataloader = DataLoader(self.sparse_interactions.coalesce().indices().t(), batch_size=10000, shuffle=False)


    def _init_graph(self, adj_mat):
        return self._convert_sp_mat_to_sp_tensor(adj_mat).to(self.device)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        initializer(self.user_embed.weight)
        initializer(self.item_embed.weight)



    # def forward(self, user, pos_item, neg_item):
    #     if user.ndim==1:
    #         user = user.unsqueeze(1)
    #     if pos_item.ndim==1:
    #         pos_item = pos_item.unsqueeze(1)
    #     if neg_item.ndim==1:
    #         neg_item = neg_item.unsqueeze(1)
    #
    #     self.user_embedding, self.item_embedding = self.gcn(self.user_embed.weight,
    #                                           self.item_embed.weight,
    #                                           edge_dropout=False,
    #                                           mess_dropout=0.5)  #original 0.2
    #
    #     self.batch_embeds = [self.user_embed(user), self.item_embed(pos_item), self.item_embed(neg_item)]
    #     u_e = self.user_tower(user)
    #     pos_e = self.item_tower(pos_item)
    #     neg_e = self.item_tower(neg_item)
    #     return {"embeds": self.batch_embeds,
    #             "user_vec": u_e, "pos_item_vec": pos_e, "neg_item_vec": neg_e}
    """version 1"""
    # def forward(self, user, pos_item, neg_item, neighbor, cate_neigbor):
    #     self.user_embedding, self.item_embedding = self.propagate(
    #         self.mat,
    #         self.user_embed.weight,
    #         self.item_embed.weight,
    #         mess_dropout_rate=self.mess_dropout_rate)
    #
    #     diverse_user_embedding = torch.zeros((user.shape[0], self.emb_size)).to(self.device)
    #     for idx, iid in enumerate(neighbor):
    #         neighbor_emb = self.item_embedding[iid]
    #         x = self.user_embedding[user][idx] - neighbor_emb
    #         dis = LA.vector_norm(x, ord=2, dim=1)
    #         #attention_score = F.softmax(dis)
    #         #print(attention_score)
    #         ####acc_attention_score = F.softmax(1 - dis)
    #         ##### self.user_embedding[user][idx] = torch.sum(neighbor_emb * acc_attention_score.reshape(-1, 1), dim=0)
    #         diverse_user_embedding[idx] = torch.sum(neighbor_emb * dis.reshape(-1, 1),dim=0)/self.emb_size  ####01.10注释
    #
    #     user_e = self.user_tower(user)
    #     pos_e = self.item_tower(pos_item)
    #     neg_e = self.item_tower(neg_item)
    #
    #     self.batch_embeds = [self.user_embed(user), self.item_embed(pos_item), self.item_embed(neg_item)]
    #     diverse_mean = torch.sum(torch.abs(user_e - diverse_user_embedding), dim=1)/self.emb_size
    #
    #     return {"embeds": self.batch_embeds, 'weight': diverse_mean,
    #             "user_vec": user_e, "user_diverse": diverse_user_embedding, "pos_item_vec": pos_e, "neg_item_vec": neg_e}

    # def fill_neighbor_list(self, input_list, fillvalue):
    #     my_len = [len(k) for k in input_list]
    #     max_len = max(my_len)
    #     result = []
    #     for my_list in input_list:
    #         if len(my_list) < max_len:
    #             for i in range(max_len - len(my_list)):
    #                 my_list.append(fillvalue)
    #         result.append(my_list)
    #     return my_len, max_len, result


    # """version 2"""
    # def forward(self, user, pos_item, neg_item, neighbor=None, cate_neigbor=None):
    #     user_propagate, self.item_embedding = self.propagate2(
    #         self.mat,
    #         self.user_embed.weight,
    #         self.item_embed.weight,
    #         mess_dropout_rate=self.mess_dropout_rate)
    #     # user_propagate = self.user_embedding.clone()
    #     user_mean = user_propagate[user]
    #
    #     pdist = nn.PairwiseDistance(p=2)
    #     score = torch.tensor([]).to(self.device)
    #     # diverse_user_embedding = torch.zeros((user.shape[0], self.emb_size)).to(self.device)
    #     # for idx, iid in enumerate(neighbor):
    #     for batch_ndx, sample in enumerate(self.cf_dataloader):
    #         index, item = sample.t()   #item, user
    #         score = torch.cat([score, pdist(self.item_embedding[item], user_propagate[index])])
    #     score_mat = torch.sparse.FloatTensor(torch.LongTensor([self.index, self.item__]).to(self.device),
    #                                          score, size=self.sparse_interactions.size())
    #     soft_score_mat = torch.sparse.softmax(score_mat, dim=1)
    #     del score_mat
    #     self.user_embedding = torch.sparse.mm(soft_score_mat, self.item_embedding)
    #     del soft_score_mat
    #     # self.user_embedding = user_agg   # diversified user embedding
    #
    #     user_e = self.user_tower(user)
    #     pos_e = self.item_tower(pos_item)
    #     neg_e = self.item_tower(neg_item)
    #
    #     self.batch_embeds = [self.user_embed(user), self.item_embed(pos_item), self.item_embed(neg_item)]
    #     diverse_mean = torch.sum(torch.abs(user_e - user_mean), dim=1)   #/ self.emb_size
    #     diverse_mean = torch.tanh(diverse_mean)
    #
    #     return {"embeds": self.batch_embeds, 'weight': diverse_mean,
    #             "user_vec": user_mean, "user_diverse": user_e, "pos_item_vec": pos_e, "neg_item_vec": neg_e}

    """version 3"""

    def forward(self, user, pos_item, neg_item, neighbor=None, cate_neigbor=None):
        self.user_embedding, self.item_embedding = self.propagate2(
            self.mat,
            self.user_embed.weight,
            self.item_embed.weight,
            mess_dropout_rate=self.mess_dropout_rate)


        pdist = nn.PairwiseDistance(p=2)
        score = torch.tensor([]).to(self.device)

        for batch_ndx, sample in enumerate(self.cf_dataloader):
            index, item = sample.t()  # item, user
            score = torch.cat([score, pdist(self.item_embedding[item], self.user_embedding[index])])
        score_mat = torch.sparse.FloatTensor(torch.LongTensor([self.index, self.item__]).to(self.device),
                                             score, size=self.sparse_interactions.size())
        soft_score_mat = torch.sparse.softmax(score_mat, dim=1)
        del score_mat
        user_diverse = torch.sparse.mm(soft_score_mat, self.item_embedding)[user]
        del soft_score_mat


        user_e = self.user_tower(user)
        pos_e = self.item_tower(pos_item)
        neg_e = self.item_tower(neg_item)

        self.batch_embeds = [self.user_embed(user), self.item_embed(pos_item), self.item_embed(neg_item)]
        diverse_mean = torch.sum(torch.abs(user_e - user_diverse), dim=1)  # / self.emb_size
        diverse_mean = torch.tanh(diverse_mean)

        return {"embeds": self.batch_embeds, 'weight': diverse_mean,
                "user_vec": user_e, "user_diverse": user_diverse, "pos_item_vec": pos_e, "neg_item_vec": neg_e}






    # def propagate(self, adj, user_emb, item_emb, mess_dropout_rate=0.5):
    #     ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
    #     ego_embeddings = nn.Dropout(p=mess_dropout_rate)(ego_embeddings)   ##new added
    #     all_embeddings = [ego_embeddings]
    #     for k in range(1, self.n_layers+1):
    #         if adj.is_sparse is True:
    #             ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
    #         else:
    #             ego_embeddings = torch.mm(adj, ego_embeddings)
    #         all_embeddings += [ego_embeddings]
    #     all_embeddings = torch.stack(all_embeddings, dim=1)
    #     all_embeddings = torch.mean(all_embeddings, dim=1)
    #     u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
    #     return u_g_embeddings, i_g_embeddings

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def propagate2(self, adj, user_emb, item_emb, mess_dropout_rate=0.5, edge_dropout=False):
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        ego_embeddings = nn.Dropout(p=mess_dropout_rate)(ego_embeddings)   ##new added

        all_embeddings = [ego_embeddings]
        for k in range(1, self.n_layers + 1 ):
            if adj.is_sparse is True:
                dropped_adj = self._sparse_dropout(adj, self.edge_dropout_rate) if edge_dropout else adj
                ego_embeddings = torch.sparse.mm(dropped_adj, ego_embeddings)
            else:
                ego_embeddings = torch.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def user_tower(self, input):
        input = input.type(torch.long)
        user_vec = self.user_embedding[input]
        return user_vec

    def item_tower(self, input):
        input = input.type(torch.long)
        item_vec = self.item_embedding[input]
        return item_vec

    def get_user_item_embedding(self):
        return self.user_embedding, self.item_embedding






