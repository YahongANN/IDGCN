
from src.data_generator import *
import numpy as np
import torch
import torch.nn as nn
from src.metrics import evaluate_metrics
import src.models.loss_function as loss_function
from tqdm import tqdm
import torch.nn.functional as F

class IDGCNTrainer():
    def __init__(self,
                 optimizer="Adam",
                 gpu=-1,
                 item_cate_dict=dict(),
                 new_item2cate =[],
                 sample_cate = 4,
                 metrics=None,
                 num_epochs=1000,
                 lr=None,
                 emb_lambda=None,
                 save_model_path=None,
                 is_pretrained=False,
                 is_save_embedding=False,
                 weight_decay=0,
                 cul_total_epoch=0,
                 per_eval=5,
                 model_config=None,
                 data_config=None,
                 uni_weight=0,
                 loss_temp=0.1,
                 item_cate_size=0,
                 save_emb_name=None,
                 **kwargs):
        print(kwargs)
        print("Building IDGCN Trainer...")

        self.loss_temp = loss_temp
        self.cul_total_epoch = cul_total_epoch
        self.kwargs = kwargs
        self.weight_decay = weight_decay
        self.emb_lambda = emb_lambda
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer
        self._validation_metrics = metrics
        self.save_emb_name = save_emb_name



        self.infonce = loss_function.InfoNCE(loss_temp)
        self.uni_weight = uni_weight
        print("uni_weight is: ", self.uni_weight)

        self._best_metrics = 0
        self.lr = lr
        self.save_model_path = save_model_path
        self.is_pretrained = is_pretrained
        self.is_save_embedding = is_save_embedding
        self.device = self.set_device(gpu)
        self.item_cate_num = item_cate_size
        self.item_cate_dict = item_cate_dict


        self.model_name = model_config["model_name"]
        #    build data:
        self.train_gen, self.valid_gen, self.test_gen = data_generator(model_name=self.model_name, item_cate_dict = self.item_cate_dict, new_item2cate= new_item2cate, **data_config)
        self.diverse_weight = self.train_gen.get_user_diverse()



        adj_mat = self.train_gen.adj_mat
        self.data_config = data_config

        #    build model:
        model_config["device"] = self.device
        if model_config["model_name"]=="lightgcn" or model_config["model_name"]=="algcn" \
                or model_config["model_name"]=="pdgcn" or model_config["model_name"] =='idgcn':
            model_config["adj_mat"] = adj_mat

        self.model = self.build_model(model_config).to(self.device)
        self.optimizer = self.set_optimizer(self.model)

    def train(self):

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print("************ Epoch={} start ************".format(epoch))
            #   training process:
            epoch_loss, bpr_loss, emb_loss, uniform_loss = 0, 0, 0, 0
            with tqdm(total=len(self.train_gen)) as pbar:
                for batch_index, batch_data in enumerate(self.train_gen):
                    batch_data = [x.to(self.device) for x in batch_data]
                    if self.uni_weight>0.0:
                        epoch_loss_, bpr_loss_, emb_loss_, uniform_loss_ = self._step(batch_data)
                        uniform_loss += uniform_loss_
                    else:
                        epoch_loss_, bpr_loss_, emb_loss_ = self._step(batch_data)

                    epoch_loss += epoch_loss_
                    bpr_loss += bpr_loss_
                    emb_loss += emb_loss_
                    pbar.update(1)
                    # if batch_index>2:
                    #     break

            epoch_loss = epoch_loss / (len(self.train_gen))
            bpr_loss = bpr_loss / (len(self.train_gen))
            emb_loss = emb_loss / (len(self.train_gen))
            if self.uni_weight>0.0:
                uniform_loss = uniform_loss / (len(self.train_gen))
                print("Train Loss: {:.6f}, Train BPR Loss: {:.6f}, Train Emb Loss: {:.6f}, Train NCE Loss: {:.6f}".format(epoch_loss, bpr_loss, emb_loss, uniform_loss))
            else:
                print(
                    "Train Loss: {:.6f}, Train BPR Loss: {:.6f}, Train Emb Loss: {:.6f}".format(epoch_loss, bpr_loss, emb_loss))
            res_dic =  self.evaluate(self.model, self.train_gen, self.valid_gen, self.item_cate_dict)
            if epoch>0:
               if self.check_stop(res_dic, epoch, self.model):
                   break

        print("Training finished. Testing begins!")
        saved_model = torch.load(self.save_model_path)
        final_test_dict = self.evaluate(saved_model, self.train_gen, self.test_gen, self.item_cate_dict)
        print("Final results on test data")
        print(final_test_dict)




    def _step(self, batch_data):
        model = self.model.train()
        user_id, pos_item_id, neg_item_ids = batch_data[:3]
        neg_item_id = neg_item_ids[:,0]  #(batch_size, )
        cate_neighbor = neg_item_ids[:, 1:]  #(batch_size, 2*4)

        self.optimizer.zero_grad()
        temp_neighbor = []
        for uid in user_id:
            temp_neighbor.append(self.train_gen.user2items_dict[uid.item()])
        return_dict = model.forward(user_id, pos_item_id, neg_item_id, temp_neighbor, cate_neighbor)

        user_vec = return_dict["user_vec"]     #diverse user_embedding

        user_diverse = return_dict["user_diverse"]  # mean user_embedding
        pos_item_vec = return_dict["pos_item_vec"]
        neg_item_vec = return_dict["neg_item_vec"]
        diverse__ = return_dict["weight"]





        diverse_score_pos, diverse_score_neg = self.bpr_loss(user_diverse, pos_item_vec, neg_item_vec)
        score_pos, score_neg = self.bpr_loss(user_vec, pos_item_vec, neg_item_vec)

        acc_loss = -(score_pos - score_neg).sigmoid().log()  #(55766)
        diverse_loss = -(diverse_score_pos - diverse_score_neg).sigmoid().log()

        # new_weight = torch.tanh(self.diverse_weight[user_id].to(self.device))
        # loss_train = (((1 - new_weight).unsqueeze(1) * acc_loss + new_weight.unsqueeze(1) * diverse_loss)).mean()

        # ###our methods for diverse weight
        loss_train = (((1 - diverse__).unsqueeze(1) * acc_loss + diverse__.unsqueeze(1) * diverse_loss)).mean()
        # loss_train = (1.0 * acc_loss).mean()

        emb_loss = self.get_emb_loss(return_dict["embeds"])
        emb_loss = emb_loss * self.emb_lambda

        if self.uni_weight>0:
            uniform_loss = self.infonce(user_vec.squeeze(1), pos_item_vec.squeeze(1))
            uniform_loss = uniform_loss * self.uni_weight
            loss = loss_train + emb_loss + uniform_loss
        else:
            loss = loss_train + emb_loss

        loss.backward()
        self.optimizer.step()

        if self.uni_weight>0.0:
            return loss.item(), loss_train.item(), emb_loss.item(), uniform_loss.item()
        else:
            return loss.item(), loss_train.item(), emb_loss.item()


    def bpr_loss(self, user_emb_, pos_emb, neg_emb):
        pos_score = torch.sum(user_emb_ * pos_emb, dim=1)
        neg_score = torch.sum(user_emb_ * neg_emb, dim=1)
        return pos_score, neg_score
    def evaluate(self, model, train_generator, valid_generator, item_cate_dict, k=-1):
        logging.info("**** Start Evaluation ****")
        model.eval()
        with torch.no_grad():
            user_vecs = []
            item_vecs = []

            for user_batch in valid_generator.user_loader:
                user_vec = model.user_tower(user_batch.to(self.device))
                user_vecs.extend(user_vec.data.cpu().numpy())
            for item_batch in valid_generator.item_loader:
                item_vec = model.item_tower(item_batch.to(self.device))
                item_vecs.extend(item_vec.data.cpu().numpy())
            user_vecs = np.array(user_vecs, np.float64)
            item_vecs = np.array(item_vecs, np.float64)

            if k==-1:
                valid_user2items_group = None
            else:
                valid_user2items_group = valid_generator.user2items_group_dict[k]
            val_logs = evaluate_metrics(train_generator.user2items_dict,
                                        valid_generator.user2items_dict,
                                        valid_generator.test_users,
                                        self._validation_metrics,
                                        user_embs=user_vecs,
                                        item_embs=item_vecs,
                                        valid_user2items_group=valid_user2items_group,
                                        item_cate_dict = item_cate_dict,
                                        diverse_weight= train_generator.get_user_diverse().numpy(),
                                        dislike_test_path = self.data_config["dislike_test_path"]
                                        )
        return val_logs


    def get_emb_loss(self, embs):
        loss = 0
        for emb in embs:
            loss += torch.norm(emb) ** 2
        loss /= 2.0
        # batch size
        loss /= embs[0].shape[0]
        return loss


    def get_loss_fn(self, loss):
        if loss.lower()=="CosineContrastiveLoss".lower():
            print("CosineContrastiveLoss init.")
            return loss_function.CosineContrastiveLoss(self.kwargs.get("margin",0), self.kwargs.get("negative_weight"))
        elif loss.lower()=="InfoNCELoss".lower():
            print("InfoNCELoss init.")
            return loss_function.InfoNCELoss(temp=self.lossfn_temp)
        elif loss.lower()=="InfoNCE".lower():
            print("InfoNCE init.")
            return loss_function.InfoNCE()
        elif loss.lower()=="PairwiseLogisticLoss".lower():
            print("PairwiseLogisticLoss init.")
            return loss_function.PairwiseLogisticLoss()
        elif loss.lower()=="MarginalHingeLoss".lower():
            print("MarginalHingeLoss init.")
            return loss_function.MarginalHingeLoss()
        elif loss.lower()=="GumbelLoss".lower():
            print("GumbelLoss init.")
            return loss_function.GumbelLoss()

    def set_device(self, gpu=-1):
        if gpu>=0 and torch.cuda.is_available():
            device = torch.device("cuda:"+str(gpu))
        else:
            device = torch.device("cpu")
        logging.info(device)
        return device

    def set_optimizer(self, model):
        print("using: ",self.optimizer_name)
        params = []
        for m in model.modules():

            if isinstance(m, nn.Embedding):
                params.append({"params":m.parameters(),"lr":self.lr, "weight_decay":self.weight_decay})
            elif isinstance(m, nn.Linear):
                params.append({"params": m.parameters(), "lr": self.lr*0.1, "weight_decay":self.weight_decay})

        return getattr(torch.optim, self.optimizer_name)(model.parameters(), lr = self.lr)
        #return getattr(torch.optim, self.optimizer_name)(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        logging.info("saving weight successfully.")

    def set_scheduler(self):
        import torch.optim as optim
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, min_lr=self.lr*0.1)


    def build_model(self, model_config):
        from src.models.MF import MF
        from src.models.LightGCN import LightGCN

        from src.models.IDGCN import IDGCN

        model_dic = {
            "mf" : MF,
            # "algcn" : ALGCN,
            "lightgcn": LightGCN,
            'idgcn':IDGCN
        }
        return model_dic[model_config["model_name"]](**model_config)

    def save_embedding(self, model, path):
        torch.save(model.user_embedding.weight, path + "user_embedding.pt")
        torch.save(model.item_embedding.weight, path + "item_embedding.pt")
        logging.info("Saving embedding successfully.")

    def check_stop(self, res_dic, epoch, model):
        if res_dic["Recall(k=20)"] >= self._best_metrics:   #["Recall(k=20)"]
            self._best_metrics = res_dic["Recall(k=20)"]
            self._best_res = res_dic
            self._best_epoch = epoch
            torch.save(model, self.save_model_path)
            if self.save_emb_name!=None:
                self.model.save_gcn_embeds(self.save_emb_name)
            print("New best result!")
            self.early_stop_patience = 0
        else:
            self.early_stop_patience += 1

            if self.early_stop_patience >= 10:
                print("Early stopped!")
                return True
        return False



