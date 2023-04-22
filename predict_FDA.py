import numpy as np
import pandas as pd
from my_utils import predict_DTA
import os

# candidate FDA
candidate_file = r"data_input/FDA_candidate_ligands.xlsx"
candidate_df = pd.read_excel(candidate_file)
drug_SMILES = candidate_df["SMILES"].values

# top DTI models with pretrained files
top_models_file = r"tmp_files/top_models_pretrained.csv"
tmp = pd.read_csv(top_models_file, header=None).values
drug_encodings, target_encodings = tmp[:, 0], tmp[:, 1]
del tmp

# ligand rank index
if os.path.exists(r"tmp_files/predict_FDA_scores.npy"):
    result_scores = np.load(r"tmp_files/predict_FDA_scores.npy", allow_pickle=True)
else:
    result_scores = predict_DTA(drug_encodings, target_encodings, "saved_models/saved_models_S1R_t", "saved_models/saved_models_DRD2_t",
                     "saved_models/saved_models_BIP_t", drug_SMILES)
aver_scores = np.average(result_scores, axis=0)    # shape: (target_num, ligand_num), (3, 2507)
argsort_scores = np.argsort(aver_scores, axis=1)[:, ::-1][:, :20]   # DTA值最大的前20个药物的index
sort_scores = np.array([aver_scores[0, argsort_scores[0]],
                        aver_scores[1, argsort_scores[1]],
                        aver_scores[2, argsort_scores[2]]])
# output csv
pd.DataFrame(argsort_scores).to_csv(r"output_files/ligand_rank_index.csv", index=False, header=False)
pd.DataFrame(sort_scores).to_csv(r"output_files/ligand_rank_score.csv", index=False, header=False)

# ligand rank name
ligand_rank_name = candidate_df["DATABASE_ID"].values[argsort_scores]
# output csv， 用于画桑基图
pd.DataFrame(ligand_rank_name).to_csv(r"output_files/ligand_rank_name.csv", index=False, header=False)
