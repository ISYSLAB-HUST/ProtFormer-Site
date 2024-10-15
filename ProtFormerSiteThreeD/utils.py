"""
"""
import os
import json
import torch
import logging
import torch.nn.functional as F
from minlora import merge_lora
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef, precision_recall_curve, auc


#####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('3Dpredict.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)

#####################################


def prepare_protein_data(protein_labels, protein_name, combined):

    protein_labels_list = [torch.tensor([int(bit) for bit in binary_str], dtype=torch.uint8) for binary_str in
                           protein_labels]
    max_len = max(len(seq) for seq in protein_labels_list)
    protein_labels_padded = [F.pad(label, (0, max_len - len(label)), 'constant', 0) for label in protein_labels_list]
    protein_labels = torch.stack(protein_labels_padded, dim=0)
    batch_datas = []
    name_comseq = ()
    name_comseq = name_comseq + (protein_name[0], combined[0])
    batch_datas.append(name_comseq)

    return protein_labels, batch_datas

def metrics(correct_labels, predicted_labels, predicted_scores):

    ACC = accuracy_score(correct_labels, predicted_labels)
    AUC = roc_auc_score(correct_labels, predicted_scores)
    CM = confusion_matrix(correct_labels, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2 * Pre * Rec / (Pre + Rec)
    MCC = matthews_corrcoef(correct_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(correct_labels, predicted_scores)
    PRC = auc(recall, precision)

    return ACC, AUC, Rec, Pre, F1, MCC, PRC


def load_model(model, esm_model, task, save_dir):
    model_path = os.path.join(save_dir, f'{task}.pth')
    try:
        if os.path.exists(model_path):
            model_dict = torch.load(model_path)
            model.load_state_dict(model_dict['state_dict'])
            esm_model.load_state_dict(model_dict['lora_state_dict'], strict=False)
            merge_lora(esm_model)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model path {model_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")


def save_metrics_to_json(metrics, file_path):

    """
    Save metrics to a JSON file.

    Args:
    metrics (list): The list containing the metrics for each epoch.
    file_path (str): The file path to save the JSON data.
    """
    
    try:
        with open(file_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
        logger.info(f"Metrics successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to file: {e}")

