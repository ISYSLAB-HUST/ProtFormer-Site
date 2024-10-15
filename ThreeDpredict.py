import esm
import time
import logging
import resource
import numpy as np
from functools import partial
from ProtFormerSiteThreeD.data import ThreeDProteinDataset
from ProtFormerSiteThreeD.models import SSpredictor
from ProtFormerSiteThreeD.utils import prepare_protein_data, metrics, load_model
from SaProt.utils.esm_loader import load_esm_saprot
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from minlora import add_lora, LoRAParametrization
import yaml
import argparse


#####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('predict.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)

#####################################

def get_params():
    parser = argparse.ArgumentParser('ProtFormer-Site')
    parser.add_argument("--config", type=str, help='config file')
    args, _ = parser.parse_known_args()
    return args

def eval(val_loader, model, embedding_model, batch_converter, device, criterion, num_recycle):
    embedding_model.eval()
    model.eval()
    start_time = time.time()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    all_metrics = []
    with torch.no_grad():
        for batch, item in enumerate(val_loader):
            protein_names, protein_seqs, protein_labels = item
            reprs = dict()
            protein_labels, protein_name_seq = prepare_protein_data(protein_labels, protein_names, protein_seqs)
            _, _, batch_tokens = batch_converter(protein_name_seq)
            batch_tokens = batch_tokens.to(device)
            protein_labels = protein_labels.to(device)
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.autocast(device_type=device_type):
                results = embedding_model(batch_tokens, repr_layers=[33], need_head_weights=True, return_contacts=False)
                reprs["single_repr"] = results["representations"][33][:, 1:-1]
                attentions = rearrange(results["attentions"][:, -1, :, 1:-1, 1:-1], 'b h i j -> b i j h')
                reprs["pair_repr"] = attentions
                output = model(reprs, mask=None, num_recycle=num_recycle)
                predict = output["ss2"]

                predict = rearrange(predict, 'b l c -> (b l) c')
                protein_labels = rearrange(protein_labels, 'b l -> (b l)')
                protein_labels = protein_labels.to(device).long()
                metrics_labels = protein_labels
                loss = criterion(predict, protein_labels)
                predict_probs = F.softmax(output["ss2"], dim=-1)
                predict = predict_probs.argmax(dim=-1)
                predict_probs = predict_probs[:, :, 1]
                predict = rearrange(predict, 'b l -> (b l)')

            val_loss += loss.item()
            all_preds.extend(predict.cpu().numpy())
            all_targets.extend(protein_labels.cpu().numpy())
            all_probs.extend(predict_probs.cpu().numpy().flatten())

    end_time = time.time()
    logger.info(f"run Time: {end_time - start_time}")
    acc, auc, rec, pre, f1, mcc, prc = metrics(all_targets, all_preds, all_probs)

    logger.info(
        f"Test Metrics: ACC - {acc}, AUC - {auc}, Recall - {rec}, Precision - {pre}, F1 - {f1}, MCC - {mcc}, PRC - {prc}")
 
    return val_loss / len(val_loader), {
        "ACC": acc,
        "AUC": auc,
        "Recall": rec,
        "Precision": pre,
        "F1": f1,
        "MCC": mcc,
        "PRC": prc
    }

def main(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """parser config"""
    with open(args["config"], 'r') as file:
        config = yaml.safe_load(file)

    """Initialize model"""
    logger.info("Initializing model")
    model = SSpredictor(dim=config["dim"], num_layers=config["num_layers"], n_hidden=config["n_hidden"], 
                        pair_dim=config["pair_dim"], dropout=config["dropout"])
    
    lora_config = {
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=3),
        },
    }

    model_path = "./SaProt_650M_AF2/SaProt_650M_AF2.pt"
    embedding_model, alphabet = load_esm_saprot(model_path)
    batch_converter = alphabet.get_batch_converter()
    add_lora(embedding_model, lora_config)
    model = model.to(device)
    embedding_model = embedding_model.to(device)

    """load model"""
    load_model(model, embedding_model, config['task'], save_dir=config['save_dir'])

    test_data = ThreeDProteinDataset(config["test_dataset_path"], max_len=config['max_len'])
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    start_time = time.time()
    test_loss, test_metrics = eval(test_loader, model, embedding_model, batch_converter, device, config["num_recycle"])
    end_time = time.time()
    logger.info(f"run_time:{end_time-start_time}")
    logger.info(f"Test Loss: {test_loss}")
    logger.info(f"Test Metrics: {test_metrics}")
    logger.info("Finished Predicting")

if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
