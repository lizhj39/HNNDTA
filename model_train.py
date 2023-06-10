import numpy as np
import os, time
import csv
from itertools import product
import torch

from torch import nn
import pandas as pd
import pickle

from DeepPurpose.dataset import read_file_training_dataset_drug_target_pairs
from DeepPurpose.utils import data_process, generate_config
from DeepPurpose.DTI import model_initialize
from my_utils import shuffle_txt_row_


def train(data_path: list, model_candidate_csv_file: str, shuffle_data_file: bool = True, RepeatRunTimes: int = 1,
          pretrain_path: str = None, save_model: bool = True, label_text: str = "", train_val_test: list = None):
    if train_val_test is None:
        train_val_test = [0.7, 0.1, 0.2]
    assert np.sum(np.array(train_val_test)) == 1, "ratio error."

    # candidate models(drug encoding & target encoding)
    tmp = pd.read_csv(model_candidate_csv_file, header=0).values
    drug_encodings, target_encodings = tmp[:, 0], tmp[:, 1]
    del tmp

    # drug_encodings = ['Morgan', "Daylight", "Pubchem", "rdkit_2d_normalized", "CNN", "GRU", "LSTM", "MPNN", "Transformer", "ESPF", "ErG", "DGL_GCN", "DGL_NeuralFP", "DGL_AttentiveFP"]
    # target_encodings = ["AAC", "PseudoAAC", "Conjoint_triad", "Quasi-seq", "CNN", "GRU", "LSTM", "Transformer", "ESPF"]

    for t in range(RepeatRunTimes):
        if shuffle_data_file:
            [shuffle_txt_row_(d_p) for d_p in data_path]

        if len(data_path) == 1: # single target
            X_drug, X_target, y = read_file_training_dataset_drug_target_pairs(data_path[0])
        else:                   # multi target, one target for one data_path
            X_drug, X_target, y = [], [], []
            for i, d_p in enumerate(data_path):
                tmp = read_file_training_dataset_drug_target_pairs(d_p)
                X_drug.append(tmp[0]), X_target.append(tmp[1]), y.append(tmp[2])

        # open the file in the write mode
        with open(f'model_accuracy/results_{t+1}{label_text}.csv', 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(["drug_encoding", "target_encoding", "MSE", "CI"])

        # for drug_encoding, target_encoding in product(*[drug_encodings, target_encodings]):
        for drug_encoding, target_encoding in zip(drug_encodings, target_encodings):
            rnn_Use_GRU_LSTM_drug, rnn_Use_GRU_LSTM_target = "GRU", "GRU"
            if drug_encoding in ("LSTM", "GRU"):
                rnn_Use_GRU_LSTM_drug = drug_encoding
                drug_encoding = "CNN_RNN"
            if target_encoding in ("LSTM", "GRU"):
                rnn_Use_GRU_LSTM_target = target_encoding
                target_encoding = "CNN_RNN"

            if len(data_path) == 1:
                tmp = data_process(X_drug, X_target, y,
                                   drug_encoding, target_encoding,
                                   split_method='no_split',frac=train_val_test, random_seed = 1)
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
                    train.append(tmp[:index_1]), val.append(tmp[index_1 : index_2]), test.append(tmp[index_2:])
                train, val, test = pd.concat(train), pd.concat(val), pd.concat(test)
                train, val, test = train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
            else:
                raise "no data_path provided."

            # train model
            if pretrain_path is not None:
                if drug_encoding == "CNN_RNN": drug_encoding = rnn_Use_GRU_LSTM_drug
                if target_encoding == "CNN_RNN": target_encoding = rnn_Use_GRU_LSTM_target
                config = pickle.load(open(pretrain_path + f"/{drug_encoding}_{target_encoding}/config.pkl", 'rb'))
                config['train_epoch'] = 50
                model = model_initialize(**config)
                model.load_pretrained(pretrain_path + f"/{drug_encoding}_{target_encoding}/model.pt")
            else:
                config = generate_config(drug_encoding=drug_encoding,
                                         target_encoding=target_encoding,
                                         rnn_Use_GRU_LSTM_drug=rnn_Use_GRU_LSTM_drug,
                                         rnn_Use_GRU_LSTM_target=rnn_Use_GRU_LSTM_target,
                                         train_epoch=180)
                model = model_initialize(**config)

            # try:
            model.train(train, val, test, verbose=True)
            # except:
            #     with open('errors.csv', 'a+') as f:
            #         writer = csv.writer(f)
            #         if drug_encoding == "CNN_RNN": drug_encoding = rnn_Use_GRU_LSTM_drug
            #         if target_encoding == "CNN_RNN": target_encoding = rnn_Use_GRU_LSTM_target
            #         writer.writerow([drug_encoding, target_encoding])
            #     continue

            evaluations = np.load(r"tmp_files/tmp_evaluations.npy", allow_pickle=True)
            # print(f"drug_ec: {drug_encoding}, target_ec: {target_encoding}, MSE={evaluations[0]}, CI={evaluations[1]}")

            with open(f'model_accuracy/results_{t+1}{label_text}.csv', 'a+') as f:
                # create the csv writer
                writer = csv.writer(f)
                # write a row to the csv file
                if drug_encoding == "CNN_RNN": drug_encoding = rnn_Use_GRU_LSTM_drug
                if target_encoding == "CNN_RNN": target_encoding = rnn_Use_GRU_LSTM_target
                writer.writerow([drug_encoding, target_encoding, evaluations[0], evaluations[1]])

            if save_model:
                model_save_path = f"saved_models/saved_models{label_text}/{drug_encoding}_{target_encoding}"
                model.save_model(model_save_path)
                print(f"save model successfully, path={model_save_path}")
            torch.cuda.empty_cache()
            time.sleep(1)


data_path_S1R = r"tmp_files/drug_S1R_pChEMBL_datas.txt"
data_path_DRD2 = r"tmp_files/drug_DRD2_pChEMBL_datas.txt"
data_path_BIP = r"tmp_files/drug_BIP_pChEMBL_datas.txt"
model_candidate_drug = r"tmp_files/drug_encoding_candidates.csv"
model_candidate_target = r"tmp_files/target_encoding_candidates.csv"
# model_candidate_top = r"tmp_files/top_models_pretrained.csv"
pretrain_path = f"saved_models/saved_models_pre"


# 先用所有数据预训练
train([data_path_S1R, data_path_DRD2, data_path_BIP], model_candidate_target, shuffle_data_file=False,
      pretrain_path=None, save_model=True, label_text="_pre")

# shuffle_data_file要设置为False，按顺序划分数据集，避免用到预训练中测试集的数据来训练
# train([data_path_S1R], model_candidate_drug, shuffle_data_file=False, pretrain_path=None, save_model=True, label_text="_S1R_d")
# train([data_path_DRD2], model_candidate_drug, shuffle_data_file=False, pretrain_path=None, save_model=True, label_text="_DRD2_d")

train([data_path_S1R], model_candidate_target, shuffle_data_file=False, pretrain_path=None, save_model=True, label_text="_S1R_t")
train([data_path_DRD2], model_candidate_target, shuffle_data_file=False, pretrain_path=None, save_model=True, label_text="_DRD2_t")

# 然后对各自的target分别finetune模型, 此处靶标BIP的数据集不够，所以要用预训练模型
train([data_path_BIP], model_candidate_target, shuffle_data_file=False, pretrain_path=pretrain_path, save_model=True, label_text="_BIP_t")

