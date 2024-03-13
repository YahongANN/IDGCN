import json

import pandas as pd
import yaml
import os
from src.trainer.base_trainer import *
from collections import Counter
import pickle
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
    # print(model_config)
    # exit(0)
    data_config["tau"] = model_config["tau"]

    with open(trainer_config_path, 'r') as data:
        trainer_config = yaml.load(data, Loader=yaml.FullLoader)
        trainer_config = trainer_config["base"]
    print(model_config)
    print(trainer_config)
    print(data_config)
    return {"data_config":data_config, "trainer_config":trainer_config, "model_config":model_config}


def f2csv(input_tuple, output_path):
    train_df = pd.DataFrame(input_tuple, columns=["user_id", "item_id"])
    # train_df["item_id"] = train_df["item_id"] - 1
    # train_df["user_id"] = train_df["user_id"] - 1
    train_df["label"] = 1
    train_df.to_csv(output_path, sep=",", index=False)


if __name__ == '__main__':
    seed_everything()
    set_logging()

    ####tau corresponds to alpha in paper  [0.1, 0.2,0.3,0.4,0.5]
    dataset_name = "Beauty"  # "ml-10m"
    # f_out = f"./data/{dataset_name}/my_{dataset_name.lower()}_data.pkl"
    model_name = "lightgcn"
    configs_dic = get_configs(dataset_name=dataset_name, model_name=model_name)
    data_config = configs_dic["data_config"]

    df = pd.read_csv(f"./data/{dataset_name}/item_cate_dict.csv", header=0, sep=",")
    item_cate_dict = dict(zip(df["item_id"].values, df["cate"].values))

    cate_num = data_config["cate_num"]
    trainer_config = configs_dic["trainer_config"]
    model_config = configs_dic["model_config"]

    # count_ = Counter(list(item_cate_dict.values()))
    # each_cate_num = item_num // 4
    # new_cate = [[], [], [], []]
    # new_cate_val = {0: 0, 1: 0, 2: 0, 3: 0}
    # i = 0
    #
    # sorted_ = dict(sorted(count_.items(), key=lambda x: x[1]))
    # total = 0
    # for key, value in sorted_.items():
    #     total += value
    #     if i == 3:
    #         new_cate[i].append(key)
    #         new_cate_val[i] += value
    #     elif new_cate_val[i] < each_cate_num and (value < each_cate_num / 2):
    #         new_cate[i].append(key)
    #         new_cate_val[i] += value
    #     else:
    #         i += 1
    #         if new_cate_val[i] == 0:
    #             new_cate[i].append(key)
    #             new_cate_val[i] += value
    # new_item2cate = [[], [], [], []]
    #
    # for item, cate in item_cate_dict.items():
    #     for i in range(0, 4):
    #         if cate in new_cate[i]:
    #             new_item2cate[i].append(item)
    #             break
    new_item2cate = dict()
    data_config["num_neg"] = 1
    trainer_config["gpu"] = 0
    data_config["tau"] = model_config["tau"]
    item_cate_size = int(cate_num)
    del model_config["tau"]
    model_config["n_layers"] = model_config["n_layers"]
    save_model_path = f"./save_model/{dataset_name}-{model_name}.pt"
    trainer_config["save_model_path"] = save_model_path

    trainer = BaseTrainer(**trainer_config, model_config=model_config, data_config=data_config,
                          item_cate_size=item_cate_size,
                          item_cate_dict=item_cate_dict, new_item2cate=new_item2cate, sample_cate=4)
    trainer.train()













