import yaml
import os
from src.trainer.idgcn_trainer import *

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
    trainer_config_path = "./config/idgcn_trainer_config.yaml"

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

    return {"data_config":data_config, "trainer_config":trainer_config, "model_config":model_config}


def f2csv(input_tuple, output_path):
    train_df = pd.DataFrame(input_tuple, columns=["user_id", "item_id"])
    train_df["label"] = 1
    train_df.to_csv(output_path, sep=",", index=False)


if __name__ == '__main__':
    seed_everything()
    set_logging()
    dataset_name ="ml-10m"  #choose from 'ml-10m', 'Beauty', 'Music'
    model_name = "idgcn"

    for wei in [0.05]:
        for temp in [0.1]:  #infonce temperature 0.1 constant
            configs_dic = get_configs(dataset_name = dataset_name, model_name = model_name)
            data_config = configs_dic["data_config"]

            df = pd.read_csv(data_config["item_cate_path"], header=0, sep=",")
            item_cate_dict = dict(zip(df["item_id"].values, df["cate"].values))

            cate_num = data_config["cate_num"]

            trainer_config = configs_dic["trainer_config"]
            model_config = configs_dic["model_config"]


            new_item2cate = dict()
            data_config["num_neg"] = 1
            trainer_config["gpu"] = 0
            data_config["tau"] = model_config["tau"]
            item_cate_size = data_config["cate_num"]
            del model_config["tau"]
            model_config["n_layers"] = model_config["n_layers"]
            save_model_path = f"./save_model/{dataset_name}-{model_name}-wei={wei}.pt"  # 0121,0122 lr=0.001
            trainer_config["save_model_path"] = save_model_path

            print(model_config)
            print(trainer_config)
            print(data_config)
            trainer = IDGCNTrainer(**trainer_config, model_config=model_config, data_config=data_config,
                                  uni_weight=wei, loss_temp=temp, item_cate_size=item_cate_size, item_cate_dict=item_cate_dict, new_item2cate=new_item2cate, sample_cate= 4)
            trainer.train()











