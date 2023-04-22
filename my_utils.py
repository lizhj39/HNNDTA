import random
import numpy as np
import pandas as pd
import pickle
from DeepPurpose.utils import data_process_repurpose_virtual_screening, generate_config
from DeepPurpose.DTI import model_initialize


def shuffle_txt_row_(file: str):
    lines = []
    with open(file, 'r', encoding='utf-8') as f:  # 需要打乱的原文件位置
        for line in f:
            lines.append(line)
    random.shuffle(lines)

    out = open(file, 'w')   # 覆盖原文件
    for line in lines:
        out.write(line)

    out.close()


def predict_DTA(drug_encodings, target_encodings, pretrain_path_S1R, pretrain_path_DRD2, pretrain_path_BIP, candidate_SMILES, saved_output: bool = True):
    # targets
    S1R_sequence = "MQWAVGRRWAWAALLLAVAAVLTQVVWLWLGTQSFVFQREEIAQLARQYAGLDHELAFSRLIVELRRLHPGHVLPDEELQWVFVNAGGWMGAMCLLHASLSEYVLLFGTALGSRGHSGRYWAEISDTIISGTFHQWREGTTKSEVFYPGETVVHGPGEATAVEWGPNTWMVEYGRGVIPSTLAFALADTVFSTQDFLTLFYTLRSYARGLRLELTTYLFGQDP"

    DRD2_sequence = "MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCMAVSREKALQTTTNYLIVSLAVADLLVATLVMPWVVYLEVVGEWKFSRIHCDIFVTLDVMMCTASILNLCAISIDRYTAVAMPMLYNTRYSSKRRVTVMISIVWVLSFTISCPLLFGLNNADQNECIIANPAFVVYSSIVSFYVPFIVTLLVYIKIYIVLRRRRKRVNTKRSSRAFRAHLRAPLKGNCTHPEDMKLCTVIMKSNGSFPVNRRRVEAARRAQELEMEMLSSTSPPERTRYSPIPPSHHQLTLPDPSHHGLHSTPDSPAKPEKNGHAKDHPKIAKIFEIQTMPNGKTRTSLKTMSRRKLSQQKEKKATQMLAIVLGVFIICWLPFFITHILNIHCDCNIPPVLYSAFTWLGYVNSAVNPIIYTTFNIEFRKAFLKILHC"

    BIP_sequence = "MKLSLVAAMLLLLSAARAEEEDKKEDVGTVVGIDLGTTYSCVGVFKNGRVEIIANDQGNRITPSYVAFTPEGERLIGDAAKNQLTSNPENTVFDAKRLIGRTWNDPSVQQDIKFLPFKVVEKKTKPYIQVDIGGGQTKTFAPEEISAMVLTKMKETAEAYLGKKVTHAVVTVPAYFNDAQRQATKDAGTIAGLNVMRIINEPTAAAIAYGLDKREGEKNILVFDLGGGTFDVSLLTIDNGVFEVVATNGDTHLGGEDFDQRVMEHFIKLYKKKTGKDVRKDNRAVQKLRREVEKAKRALSSQHQARIEIESFYEGEDFSETLTRAKFEELNMDLFRSTMKPVQKVLEDSDLKKSDIDEIVLVGGSTRIPKIQQLVKEFFNGKEPSRGINPDEAVAYGAAVQAGVLSGDQDTGDLVLLDVCPLTLGIETVGGVMTKLIPRNTVVPTKKSQIFSTASDNQPTVTIKVYEGERPLTKDNHLLGTFDLTGIPPAPRGVPQIEVTFEIDVNGILRVTAEDKGTGNKNKITITNDQNRLTPEEIERMVNDAEKFAEEDKKLKERIDTRNELESYAYSLKNQIGDKEKLGGKLSSEDKETMEKAVEEKIEWLESHQDADIEDFKAKKKELEEIVQPIISKLYGSAGPPPTGEEDTAEKDEL"

    result = np.array([])
    for drug_encoding, target_encoding in zip(drug_encodings, target_encodings):

        # load models
        config_S1R = pickle.load(open(f"{pretrain_path_S1R}/{drug_encoding}_{target_encoding}/config.pkl", 'rb'))
        model_S1R = model_initialize(**config_S1R)
        model_S1R.load_pretrained(f"{pretrain_path_S1R}/{drug_encoding}_{target_encoding}/model.pt")

        config_DRD2 = pickle.load(open(f"{pretrain_path_DRD2}/{drug_encoding}_{target_encoding}/config.pkl", 'rb'))
        model_DRD2 = model_initialize(**config_DRD2)
        model_DRD2.load_pretrained(f"{pretrain_path_DRD2}/{drug_encoding}_{target_encoding}/model.pt")

        config_BIP = pickle.load(open(f"{pretrain_path_BIP}/{drug_encoding}_{target_encoding}/config.pkl", 'rb'))
        model_BIP = model_initialize(**config_BIP)
        model_BIP.load_pretrained(f"{pretrain_path_BIP}/{drug_encoding}_{target_encoding}/model.pt")

        # config
        if drug_encoding in ("LSTM", "GRU"):
            drug_encoding = "CNN_RNN"
        if target_encoding in ("LSTM", "GRU"):
            target_encoding = "CNN_RNN"

        # 调用DTI.py里的DBTA.predict需要调用data_process_repurpose_virtual_screening生成df_data
        pred_df_data_S1R = data_process_repurpose_virtual_screening(X_repurpose=candidate_SMILES, target=S1R_sequence,
                                                                    drug_encoding=drug_encoding,
                                                                    target_encoding=target_encoding, mode="virtual screening")
        pred_df_data_DRD2 = data_process_repurpose_virtual_screening(X_repurpose=candidate_SMILES, target=DRD2_sequence,
                                                                     drug_encoding=drug_encoding,
                                                                     target_encoding=target_encoding, mode="virtual screening")
        pred_df_data_BIP = data_process_repurpose_virtual_screening(X_repurpose=candidate_SMILES, target=BIP_sequence,
                                                                    drug_encoding=drug_encoding,
                                                                    target_encoding=target_encoding, mode="virtual screening")

        # predict
        scores_S1R = model_S1R.predict(pred_df_data_S1R)    # list, shape:len(candidate_SMILES)
        scores_DRD2 = model_DRD2.predict(pred_df_data_DRD2)
        scores_BIP = model_BIP.predict(pred_df_data_BIP)

        tmp_np = np.expand_dims(np.array([scores_S1R, scores_DRD2, scores_BIP]), axis=0)
        if result.shape[0] == 0:
            result = tmp_np
        else:
            result = np.concatenate([result, tmp_np], axis=0)   # result shape: (model_num, target_num, ligand_num), (7, 3, 2507)

    if saved_output:
        np.save(r"tmp_files/predict_FDA_scores.npy", result)

    return result
