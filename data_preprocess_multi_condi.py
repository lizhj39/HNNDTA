import pandas as pd
import numpy as np
from pathlib import Path


def output_drug_target_pChEMBL(ligands: pd.DataFrame, target_sequence: str, target_name: str, output_txt: bool = False):
    # return: drug_target_pChEMBL_datas, ndarray, shape(*,3)
    # output: txt [optional]

    unique_prot = set(ligands.values[:,0])
    drug_target_pChEMBL_datas = []
    for prot in unique_prot:
        one_ligand_all = ligands[ligands["Molecule ChEMBL ID"]==prot]
        tmp_data = [one_ligand_all["Smiles"].values[0]]
        tmp_data.append(target_sequence)

        tmp_value = one_ligand_all["pChEMBL Value"].values
        if one_ligand_all.shape[0] >= 3:
            tmp_mean = (tmp_value.sum() - tmp_value.max() - tmp_value.min()) / (one_ligand_all.shape[0] - 2)
        else:
            tmp_mean = tmp_value.mean()
        tmp_data.append(np.around(tmp_mean, decimals=2))
        drug_target_pChEMBL_datas.append(tmp_data)

    drug_target_pChEMBL_datas = np.array(drug_target_pChEMBL_datas)
    if output_txt:
        output_txt_path = f"tmp_files/drug_{target_name}_pChEMBL_datas.txt"
        np.savetxt(output_txt_path, drug_target_pChEMBL_datas, delimiter="\t", fmt = "%s")
        print(f"generate drug_target_pChEMBL txt file sucessfully. path={output_txt_path}")
    return drug_target_pChEMBL_datas


# target sequence
S1R_sequence = "MQWAVGRRWAWAALLLAVAAVLTQVVWLWLGTQSFVFQREEIAQLARQYAGLDHELAFSRLIVELRRLHPGHVLPDEELQWVFVNAGGWMGAMCLLHASLSEYVLLFGTALGSRGHSGRYWAEISDTIISGTFHQWREGTTKSEVFYPGETVVHGPGEATAVEWGPNTWMVEYGRGVIPSTLAFALADTVFSTQDFLTLFYTLRSYARGLRLELTTYLFGQDP"

DRD2_sequence = "MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVSREKALQTTTNYLIVSLAVADLLVATLVMPWVVYLEVVGEWKFSRIHCDIFVTLDVMMCTASILNLCAISIDRYTAVAMPMLYNTRYSSKRRVTVMISIVWVLSFTISCPLLFGLNNADQNECIIANPAFVVYSSIVSFYVPFIVTLLVYIKIYIVLRRRRKRVNTKRSSRAFRAHLRAPLKGNCTHPEDMKLCTVIMKSNGSFPVNRRRVEAARRAQELEMEMLSSTSPPERTRYSPIPPSHHQLTLPDPSHHGLHSTPDSPAKPEKNGHAKDHPKIAKIFEIQTMPNGKTRTSLKTMSRRKLSQQKEKKATQMLAIVLGVFIICWLPFFITHILNIHCDCNIPPVLYSAFTWLGYVNSAVNPIIYTTFNIEFRKAFLKILHC"

BIP_sequence = "MKLSLVAAMLLLLSAARAEEEDKKEDVGTVVGIDLGTTYSCVGVFKNGRVEIIANDQGNRITPSYVAFTPEGERLIGDAAKNQLTSNPENTVFDAKRLIGRTWNDPSVQQDIKFLPFKVVEKKTKPYIQVDIGGGQTKTFAPEEISAMVLTKMKETAEAYLGKKVTHAVVTVPAYFNDAQRQATKDAGTIAGLNVMRIINEPTAAAIAYGLDKREGEKNILVFDLGGGTFDVSLLTIDNGVFEVVATNGDTHLGGEDFDQRVMEHFIKLYKKKTGKDVRKDNRAVQKLRREVEKAKRALSSQHQARIEIESFYEGEDFSETLTRAKFEELNMDLFRSTMKPVQKVLEDSDLKKSDIDEIVLVGGSTRIPKIQQLVKEFFNGKEPSRGINPDEAVAYGAAVQAGVLSGDQDTGDLVLLDVCPLTLGIETVGGVMTKLIPRNTVVPTKKSQIFSTASDNQPTVTIKVYEGERPLTKDNHLLGTFDLTGIPPAPRGVPQIEVTFEIDVNGILRVTAEDKGTGNKNKITITNDQNRLTPEEIERMVNDAEKFAEEDKKLKERIDTRNELESYAYSLKNQIGDKEKLGGKLSSEDKETMEKAVEEKIEWLESHQDADIEDFKAKKKELEEIVQPIISKLYGSAGPPPTGEEDTAEKDEL"

# downloaded csv files from ChEMBL
# 注意需要另存为以逗号分隔的csv
input_path_S1R = r"data_input/S1R_ligands.csv"
input_path_DRD2 = r"data_input/DRD2_ligands.csv"
input_path_BIP = r"data_input/BIP_ligands.csv"

ligands_S1R = pd.read_csv(input_path_S1R)
ligands_DRD2 = pd.read_csv(input_path_DRD2)
ligands_BIP = pd.read_csv(input_path_BIP)

# 生成 drug-target-pChEMBL 的txt文件和ndarray
drug_target_pChEMBL_S1R =  output_drug_target_pChEMBL(ligands_S1R, S1R_sequence, "S1R", output_txt=True)
drug_target_pChEMBL_DRD2 = output_drug_target_pChEMBL(ligands_DRD2, DRD2_sequence, "DRD2", output_txt=True)
drug_target_pChEMBL_BIP =  output_drug_target_pChEMBL(ligands_BIP, BIP_sequence, "BIP", output_txt=True)

# 合并上述几个target的drug-target-pChEMBL，并输出txt文件
# drug_target_pChEMBL_final = np.concatenate([drug_target_pChEMBL_S1R, drug_target_pChEMBL_DRD2, drug_target_pChEMBL_BIP], axis=0)
# np.savetxt(r"data_output/drug_target_pChEMBL.txt", drug_target_pChEMBL_final, delimiter="\t", fmt = "%s")


print('OK')
