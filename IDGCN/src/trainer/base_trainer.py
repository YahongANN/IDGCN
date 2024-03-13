
from src.data_generator import *
import numpy as np
import torch
import torch.nn as nn
from src.metrics import evaluate_metrics
import src.models.loss_function as loss_function
from tqdm import tqdm
import torch.nn.functional as F

class BaseTrainer():
    def __init__(self,
                 optimizer="Adam",
                 gpu=-1,
                 metrics=None,
                 num_epochs=1000,
                 lr=None,
                 emb_lambda=None,
                 loss=None,
                 save_model_path=None,
                 is_pretrained=False,
                 is_save_embedding=False,
                 weight_decay=0,
                 cul_total_epoch=0,
                 per_eval=5,
                 model_config=None,
                 data_config=None,
                 loss_temp=0.1,
                 uni_weight=0,
                 item_cate_size=0,
                 item_cate_dict=dict(),
                 new_item2cate=[],
                 sample_cate=4,
                 save_emb_name=None,
                 **kwargs):
        print(kwargs)
        print("Building Base Trainer...")

        self.loss_temp = loss_temp
        self.cul_total_epoch = cul_total_epoch
        self.kwargs = kwargs
        self.weight_decay = weight_decay
        self.emb_lambda = emb_lambda
        self.num_epochs = num_epochs
        self.optimizer_name = optimizer
        self._validation_metrics = metrics
        self.save_emb_name = save_emb_name

        #   build loss function:


        self.loss_fn = self.get_loss_fn(loss)  #"PairwiseLogisticLoss"
        self.infonce = loss_function.InfoNCE(self.loss_temp)
        #self.uniform_loss = loss_function.UniformLoss(1)
        #self.directAU = loss_function.DirectAU(gamma)
        self.uni_weight = uni_weight

        self.per_eval = per_eval
        self._best_metrics = 0

        self._best_val_loss = 9999
        self.lr = lr
        self.save_model_path = save_model_path

        self.is_pretrained = is_pretrained             #False
        self.is_save_embedding = is_save_embedding     #False
        self.device = self.set_device(gpu)
        self.item_cate_num = item_cate_size

        self.item_cate_dict = item_cate_dict
        self.model_name = model_config["model_name"]

        #    build data:
        self.train_gen, self.valid_gen, self.test_gen = data_generator(model_name=self.model_name, item_cate_dict = self.item_cate_dict,new_item2cate= new_item2cate, **data_config, mode="")
        adj_mat = self.train_gen.adj_mat  #adjecent matrix
        self.diverse_weight = self.train_gen.get_user_diverse()


        #=========build model=========
        model_config["device"] = self.device
        if model_config["model_name"]=="lightgcn" or model_config["model_name"]=="algcn" or model_config["model_name"] =="pdgcn":
            model_config["adj_mat"] = adj_mat
        self.model = self.build_model(model_config).to(self.device)

        print(self.model)

        self.optimizer = self.set_optimizer(self.model)


    def train(self):
        if self.model_name=="algcn" or self.model_name=="pdgcn":
            self.model.preprocess()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            print("************ Epoch={} start ************".format(epoch))
            #   training process:
            epoch_loss, bpr_loss, emb_loss, uniform_loss = 0,0,0,0
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
            epoch_loss = epoch_loss / (len(self.train_gen))
            bpr_loss = bpr_loss / (len(self.train_gen))
            emb_loss = emb_loss / (len(self.train_gen))

            if self.uni_weight>0.0:
                uniform_loss = uniform_loss / (len(self.train_gen))
                print("Train Loss: {:.6f}, Train BPR Loss: {:.6f}, Train Emb Loss: {:.6f}, Train NCE Loss: {:.6f}".format(
                epoch_loss, bpr_loss, emb_loss, uniform_loss))
            else:
                print("Train Loss: {:.6f}, Train BPR Loss: {:.6f}, Train Emb Loss: {:.6f}".format(
                        epoch_loss, bpr_loss, emb_loss))

            res_val_dict = self.evaluate(self.model, self.train_gen, self.valid_gen, self.item_cate_dict)



            print(res_val_dict)
            if epoch>0 and self.check_stop(res_val_dict, epoch, self.model):
                print("Early stopping!")
                break


        print("Final testing start")
        saved_model = torch.load(self.save_model_path)
        final_res_dic = self.evaluate(saved_model, self.train_gen, self.test_gen, self.item_cate_dict)
        print("Final results on test data")
        print(final_res_dic)

        print("Training finished")



    def bpr_loss(self, user_emb_, pos_emb, neg_emb):
        pos_score = torch.sum(user_emb_ * pos_emb, dim=1)
        neg_score = torch.sum(user_emb_ * neg_emb, dim=1)
        return pos_score, neg_score

    def _step(self, batch_data):
        model = self.model.train()
        user_id, pos_item_id, neg_item_id = batch_data[:3]
        neg_item_id = neg_item_id.squeeze(1)


        self.optimizer.zero_grad()

        return_dict = model.forward(user_id, pos_item_id, neg_item_id)

        user_vec = return_dict["user_vec"]
        pos_item_vec = return_dict["pos_item_vec"]
        neg_item_vec = return_dict["neg_item_vec"]

        if self.model_name=="mf":
            pos_y_pred = return_dict["pos_y_pred"]
            neg_y_pred = return_dict["neg_y_pred"]
        elif self.model_name=="lightgcn":
            pos_y_pred = torch.sum(user_vec * pos_item_vec, dim=1)
            neg_y_pred = torch.sum(user_vec* neg_item_vec, dim=1)


        bpr_loss = self.loss_fn(pos_y_pred, neg_y_pred)
        emb_loss = self.get_emb_loss(return_dict["embeds"])
        emb_loss = emb_loss * self.emb_lambda
        if self.uni_weight>0:   #for algcn model
            uniform_loss = self.infonce(user_vec.squeeze(1), pos_item_vec.squeeze(1))
            uniform_loss = uniform_loss * self.uni_weight
            loss = bpr_loss +  emb_loss  +  uniform_loss
        else:    #for lightgcn model
            loss = bpr_loss + emb_loss
        loss.backward()
        self.optimizer.step()

        if self.uni_weight>0:
            return loss.item(), bpr_loss.item(), emb_loss.item(), uniform_loss.item()
        else:
            return loss.item(), bpr_loss.item(), emb_loss.item()

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
                                        diverse_weight = train_generator.get_user_diverse().numpy()
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
        elif loss.lower()=="BPRLoss".lower():
            print("BPRLoss init.")
            return  loss_function.BPRLoss()
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
        # from src.models.ALGCN import ALGCN
        from src.models.LightGCN import LightGCN

        model_dic = {
            "mf" : MF,
            # "algcn" : ALGCN,
            "lightgcn": LightGCN
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



