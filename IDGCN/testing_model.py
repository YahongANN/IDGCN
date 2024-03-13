import numpy as np
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
from src.trainer.idgcn_trainer import *
import pandas as pd
from src.metrics import  evaluate_metrics, evaluate_metrics_individual


def set_logging():
    logging.basicConfig(level="INFO")


def seed_everything(seed=2020):
    logging.info("set seed as {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs(dataset_name=None, model_name=None):
    print("dataset:", dataset_name)
    data_config_path = "./config/data_config/" + dataset_name + ".yaml"
    model_config_path = "./config/model_config.yaml"
    trainer_config_path = "./config/trainer_config.yaml"

    with open(data_config_path, 'r') as data:
        data_config = yaml.load(data, Loader=yaml.FullLoader)
        data_config = data_config["base"]


    with open(model_config_path, 'r') as data:
        model_config = yaml.load(data, Loader=yaml.FullLoader)
        model_config = model_config[dataset_name + "-" + model_name]
    model_config["max_user_id"] = data_config["max_user_id"]
    model_config["max_item_id"] = data_config["max_item_id"]

    data_config["tau"] = model_config["tau"]

    with open(trainer_config_path, 'r') as data:
        trainer_config = yaml.load(data, Loader=yaml.FullLoader)
        trainer_config = trainer_config["base"]
    print(model_config)
    print(trainer_config)
    print(data_config)
    return {"data_config":data_config, "trainer_config":trainer_config, "model_config":model_config}

def evaluate_origianl(dataset_name, model, train_generator, valid_generator, item_cate_dict, k=-1):
    logging.info("**** Start Evaluation ****")
    model.eval()
    device = "cuda:0"
    _validation_metrics = ['Recall(k=20)', 'Recall(k=40)', 'NDCG(k=20)', 'NDCG(k=40)', 'HitRate(k=20)', 'HitRate(k=40)',
    'Coverage(k=20)', 'Coverage(k=40)', 'IHC(k=20)', 'IHC(k=40)']
    with torch.no_grad():
        user_vecs = []
        item_vecs = []
        for user_batch in valid_generator.user_loader:
            user_vec = model.user_tower(user_batch.to(device))
            user_vecs.extend(user_vec.data.cpu().numpy())
        for item_batch in valid_generator.item_loader:
            item_vec = model.item_tower(item_batch.to(device))
            item_vecs.extend(item_vec.data.cpu().numpy())
        user_vecs = np.array(user_vecs, np.float64)
        item_vecs = np.array(item_vecs, np.float64)

        if k == -1:
            valid_user2items_group = None
        else:
            valid_user2items_group = valid_generator.user2items_group_dict[k]
        val_logs = evaluate_metrics(train_generator.user2items_dict,
                                    valid_generator.user2items_dict,
                                    valid_generator.test_users,
                                    _validation_metrics,
                                    user_embs=user_vecs,
                                    item_embs=item_vecs,
                                    valid_user2items_group=valid_user2items_group,
                                    item_cate_dict=item_cate_dict,
                                    diverse_weight=train_generator.get_user_diverse().numpy()
                                    )
    return val_logs



if __name__ == '__main__':
    seed_everything()
    set_logging()
    dataset_name ="ml-10m"  #Beauty, ml-10m,Music
    model_name = "idgcn"

    for wei in [0.05]:
        for temp in [0.1]:  #infonce temperature 0.1 constant
            configs_dic = get_configs(dataset_name = dataset_name, model_name = model_name); data_config = configs_dic["data_config"];
            df = pd.read_csv(f"./data/{dataset_name}/item_cate_dict.csv", header=0, sep=",");
            item_cate_dict = dict(zip(df["item_id"].values, df["cate"].values));
            cate_num = data_config["cate_num"];
            trainer_config = configs_dic["trainer_config"];
            model_config = configs_dic["model_config"];
            new_item2cate = dict();
            data_config["num_neg"] = 1;
            trainer_config["gpu"] = 0;
            data_config["tau"] = model_config["tau"];
            item_cate_size = data_config["cate_num"]
            del model_config["tau"];
            if model_name !="mf":
                model_config["n_layers"] = model_config["n_layers"];
            save_model_path = f"./save_model/{dataset_name}-{model_name}-wei={wei}-0127.pt";
            train_gen, valid_gen, test_gen = data_generator(model_name=model_name,item_cate_dict=item_cate_dict, new_item2cate=new_item2cate, **data_config);
            saved_model = torch.load(save_model_path)
            evaluate_origianl(dataset_name, saved_model, train_gen, test_gen, item_cate_dict=item_cate_dict)