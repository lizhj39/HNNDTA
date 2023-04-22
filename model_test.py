import pandas as pd
import numpy as np
import pickle
from torch.utils import data
from torch.utils.data import SequentialSampler
from DeepPurpose.dataset import read_file_training_dataset_drug_target_pairs
from DeepPurpose.utils import data_process, generate_config, data_process_loader, mpnn_collate_func
from DeepPurpose.DTI import model_initialize, dgl_collate_func


data_path_S1R = r"tmp_files/drug_S1R_pChEMBL_datas.txt"
data_path_DRD2 = r"tmp_files/drug_DRD2_pChEMBL_datas.txt"
data_path_BIP = r"tmp_files/drug_BIP_pChEMBL_datas.txt"
model_candidate_top = r"tmp_files/top_models_pretrained.csv"
saved_path_S1R = f"saved_models/saved_models_S1R_t"
saved_path_DRD2 = f"saved_models/saved_models_DRD2_t"


def test(data_path: list, model_candidate_csv_file: str, saved_model_path: str = None,
         train_val_test: list = None, label_text: str = None, output_label_pred_path: str = None,
         generate_ensemble_dataset: bool = False):
    # 参数说明：
    # generate_ensemble_dataset - 是否生成集成学习的数据集。每个单一模型的预测结果作为集成学习算法的输入，真实值作为集成学习的label

    if train_val_test is None:
        train_val_test = [0.7, 0.1, 0.2]
    assert np.sum(np.array(train_val_test)) == 1, "ratio error."

    # 读取top模型
    tmp = pd.read_csv(model_candidate_csv_file, header=None).values
    drug_encodings, target_encodings = tmp[:, 0], tmp[:, 1]
    del tmp

    # 读取drug_target_label数据
    if len(data_path) == 1:  # single target
        X_drug, X_target, y = read_file_training_dataset_drug_target_pairs(data_path[0])
    else:  # multi target, one target for one data_path
        X_drug, X_target, y = [], [], []
        for i, d_p in enumerate(data_path):
            tmp = read_file_training_dataset_drug_target_pairs(d_p)
            X_drug.append(tmp[0]), X_target.append(tmp[1]), y.append(tmp[2])

    # 输出的label, pred  csv文件
    label_pred = pd.DataFrame()

    # for drug_encoding, target_encoding in product(*[drug_encodings, target_encodings]):
    for drug_encoding, target_encoding in zip(drug_encodings, target_encodings):
        rnn_Use_GRU_LSTM_drug, rnn_Use_GRU_LSTM_target = "GRU", "GRU"
        if drug_encoding in ("LSTM", "GRU"):
            rnn_Use_GRU_LSTM_drug = drug_encoding
            drug_encoding = "CNN_RNN"
        if target_encoding in ("LSTM", "GRU"):
            rnn_Use_GRU_LSTM_target = target_encoding
            target_encoding = "CNN_RNN"

        # 划分train, val, test
        if len(data_path) == 1:
            tmp = data_process(X_drug, X_target, y,
                               drug_encoding, target_encoding,
                               split_method='no_split', frac=train_val_test, random_seed=1)
            if generate_ensemble_dataset:
                e_train = tmp
            # 按顺序分割train, val, test，以便后续训练仍使用"train"中的数据
            index_1 = int(tmp.shape[0] * train_val_test[0])
            index_2 = int(tmp.shape[0] * (1 - train_val_test[-1]))
            train, val, test = tmp[:index_1], tmp[index_1: index_2], tmp[index_2:]
            train, val, test = train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
        elif len(data_path) > 1:
            train, val, test = [], [], []
            for i in range(len(data_path)):
                tmp = data_process(X_drug[i], X_target[i], y[i],
                                   drug_encoding, target_encoding,
                                   split_method='no_split',
                                   frac=train_val_test, random_seed=1)
                # 按顺序分割train, val, test，以便后续训练仍使用"train"中的数据
                index_1 = int(tmp.shape[0] * train_val_test[0])
                index_2 = int(tmp.shape[0] * (1 - train_val_test[-1]))
                train.append(tmp[:index_1]), val.append(tmp[index_1: index_2]), test.append(tmp[index_2:])
            train, val, test = pd.concat(train), pd.concat(val), pd.concat(test)
            train, val, test = train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
        else:
            raise "no data_path provided."

        # 加载模型
        if saved_model_path is not None:
            if drug_encoding == "CNN_RNN": drug_encoding = rnn_Use_GRU_LSTM_drug
            if target_encoding == "CNN_RNN": target_encoding = rnn_Use_GRU_LSTM_target
            config = pickle.load(open(saved_model_path + f"/{drug_encoding}_{target_encoding}/config.pkl", 'rb'))
            config['train_epoch'] = 50
            model = model_initialize(**config)
            model.load_pretrained(saved_model_path + f"/{drug_encoding}_{target_encoding}/model.pt")
        else:
            config = generate_config(drug_encoding=drug_encoding,
                                     target_encoding=target_encoding,
                                     rnn_Use_GRU_LSTM_drug=rnn_Use_GRU_LSTM_drug,
                                     rnn_Use_GRU_LSTM_target=rnn_Use_GRU_LSTM_target,
                                     train_epoch=180)
            model = model_initialize(**config)

        # 生成test_loader
        if generate_ensemble_dataset:
            test = e_train
        info = data_process_loader(test.index.values, test.Label.values, test, **config)
        params_test = {'batch_size': config['batch_size'],
                       'shuffle': False,
                       'num_workers': config['num_workers'],
                       'drop_last': False,
                       'sampler': SequentialSampler(info)}
        if (drug_encoding == "MPNN"):
            params_test['collate_fn'] = mpnn_collate_func
        elif drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred',
                                    'DGL_AttentiveFP']:
            params_test['collate_fn'] = dgl_collate_func
        testing_generator = data.DataLoader(
            data_process_loader(test.index.values, test.Label.values, test, **config), **params_test)

        mse, r2, p_val, CI, logits, y_label, loss_test = model.test_(testing_generator, model.model.to(model.device))

        if "label" not in label_pred.columns:
            label_pred["label"] = y_label
        label_pred[f"{drug_encoding}_{target_encoding}"] = logits

    label_pred['average'] = label_pred.iloc[:,1:].mean(axis=1)
    label_pred.to_csv(f"{output_label_pred_path}/label_pred{label_text}.csv", sep=",", index=False)


test([data_path_S1R], model_candidate_top, saved_path_S1R, label_text="_S1R", output_label_pred_path="output_files")
# test([data_path_S1R], model_candidate_top, saved_path_S1R, label_text="_S1R_EL", output_label_pred_path="output_files", generate_ensemble_dataset=True)
# test([data_path_DRD2], model_candidate_top, saved_path_DRD2, label_text="_DRD2")
