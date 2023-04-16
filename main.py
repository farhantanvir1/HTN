#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:58:38 2021

@author: illusionist
"""

# main.py
from model_hetero import GAT, GATLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    y_predict = indices.long().cpu().numpy()
    y_test = labels.cpu().numpy()

    accuracy = (y_predict == y_test).sum() / len(y_predict)
    micro_f1 = f1_score(y_test, y_predict, average='micro')
    macro_f1 = f1_score(y_test, y_predict, average='macro')
    rec = recall_score(y_test, y_predict, average='macro')
    prc = precision_score(y_test, y_predict, average='macro')

    return accuracy, micro_f1, macro_f1, rec, prc 

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1, rec, prc = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1, rec, prc

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU
    
    graph_data, train_pos_drug, train_pos_target, train_pos_disease, test_pos_drug, test_pos_target, test_pos_disease, train_neg_drug, train_neg_target, train_neg_disease, test_neg_drug, test_neg_target, test_neg_disease =. load_drug_data(devide)
    # Model architecture related - this is the architecture as defined in the official paper (for Cora classification)
    gat_config = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [32, 8, 2],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout
        "lr":0.005,
        "weight_decay": 0.001,
        "num_of_epochs": 10000,
        "patience": 1000
    }

    config = {
        'dataset_name': 'DrugBank'
    }

    # Step 2: prepare the model
    gat = GAT(
            num_of_layers=gat_config['num_of_layers'],
            num_heads_per_layer=gat_config['num_heads_per_layer'],
            num_features_per_layer=gat_config['num_features_per_layer'],
            add_skip_connection=gat_config['add_skip_connection'],
            bias=gat_config['bias'],
            dropout=gat_config['dropout'],
            log_attention_weights=False  # no need to store attentions, used only in playground.py while visualizing
        ).to(device)

    # Step 3: Prepare other training related utilities (loss & optimizer and decorator function)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(gat.parameters(), lr=gat_config["lr"], weight_decay=gat_config["weight_decay"])

    for e in range(args["num_of_epochs"]):
        # forward
        h= gat(graph_data)
        print(h)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

        # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", type=bool, help='should test the model on the test dataset?', default=True)

    args = parser.parse_args("")

    main(args)