#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:58:38 2021

@author: illusionist
"""

# main.py
import argparse
from model_hetero import HTN, HTNLayer
from util import EarlyStopping, load_drug_data
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import argparse
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torch.utils.data import DataLoader, TensorDataset
import time

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    device = torch.device("cpu" if torch.cuda.is_available() else "cuda")  # checking whether you have a GPU

    # Load data and prepare the model
    drug_target_disease_list, neg_drug_target_disease_list, all_node_features, hetero_graph, drug_id, target_id, disease_id = load_drug_data(device)

    # Combine positive and negative samples
    drug_target_disease_tensor = torch.tensor(drug_target_disease_list, dtype=torch.long).view(-1, 3).to(device)
    neg_drug_target_disease_tensor = torch.tensor(neg_drug_target_disease_list, dtype=torch.long).view(-1, 3).to(device)

    # Concatenate positive and negative samples
    combined_drug_target_disease_tensor = torch.cat((drug_target_disease_tensor, neg_drug_target_disease_tensor), dim=0)

    # Create labels for positive and negative samples
    positive_labels = torch.ones(drug_target_disease_tensor.size(0), 1)
    negative_labels = torch.zeros(neg_drug_target_disease_tensor.size(0), 1)

    # Concatenate labels for positive and negative samples
    combined_labels = torch.cat((positive_labels, negative_labels), dim=0)
    labels_np = combined_labels.numpy()
    if np.isnan(labels_np).any() or np.isinf(labels_np).any():
      print("Warning: Labels contain NaN or infinite values.")

    # Split combined tensor and labels into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(combined_drug_target_disease_tensor, combined_labels, test_size=0.2, random_state=42)

    # Create TensorDatasets for train and test sets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    # Instantiate the GNN model
    model = HTN(args["num_of_layers"], args["num_heads_per_layer"], args["num_features_per_layer"]).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    time_start = time.time()

    for epoch in range(args["num_of_epochs"]):
      model.train()
      epoch_loss = 0.0

      for batch in train_loader:
          drug_target_disease_batch, y_batch = batch
          drug_target_disease_batch, y_batch = drug_target_disease_batch.to(device), y_batch.to(device)

          labels_np = drug_target_disease_batch.numpy()
          if np.isnan(labels_np).any() or np.isinf(labels_np).any():
            print("Warning: Labels contain NaN or infinite values.")

          hetero_graph_np = hetero_graph.x.numpy()
          if np.isnan(hetero_graph_np).any() or np.isinf(hetero_graph_np).any():
            print("Warning: hetero_graph_np contain NaN or infinite values.")

          y_batch_np = y_batch.numpy()
          if np.isnan(y_batch_np).any() or np.isinf(y_batch_np).any():
            print("Warning: y_batch_np contain NaN or infinite values.")
          
          optimizer.zero_grad()

          # Forward pass
          out = model(hetero_graph.x, drug_target_disease_batch)

          # Get the predicted interaction values
          y_pred = model.forward_predictor(out, drug_target_disease_batch[:, 0], drug_target_disease_batch[:, 1], drug_target_disease_batch[:, 2])

          # Calculate loss
          loss = criterion(y_pred, y_batch)

          # Backward pass and optimization
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()

          epoch_loss += loss.item()

      # Print average loss for the current epoch 
      print(f'Graph Triplet Attention Network training: time elapsed= {(time.time() - time_start):.2f} [s] | Epoch= {epoch + 1} | Loss= {epoch_loss/ len(X_train)}')

    # Print average loss for the current epoch
    #print(f'HTN Epoch: {epoch + 1}, Loss: {epoch_loss / len(X_train)}')

    model.eval()
    correct = 0
    total = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            drug_target_disease_batch, labels = batch
            drug_target_disease_batch, labels = drug_target_disease_batch.to(device), labels.to(device)

            # Forward pass
            out = model(hetero_graph.x, drug_target_disease_batch)

            # Get the predicted interaction values
            y_pred = model.forward_predictor(out, drug_target_disease_batch[:, 0], drug_target_disease_batch[:, 1], drug_target_disease_batch[:, 2])

            # Apply a threshold to obtain binary predictions
            binary_predictions = (y_pred > 0.5).float()
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(y_pred.cpu().numpy())
            correct += (binary_predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    # Convert lists to numpy arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Compute ROC AUC score
    roc_auc = roc_auc_score(y_true, y_pred)

    # Compute AUPR score
    aupr = average_precision_score(y_true, y_pred)

    print(f"ROC AUC: {roc_auc* 100:.2f}%")
    print(f"AUPR: {aupr* 100:.2f}%")
    print(f'Test accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    from util import setup

    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=10000)
    parser.add_argument("--patience_period", type=int, help="number of epochs with no improvement on val before terminating", default=1000)
    parser.add_argument("--lr", type=float, help="model learning rate", default=5e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularization on model weights", default=5e-4)
    parser.add_argument("--should_test", type=bool, help='should test the model on the test dataset?', default=True)

    #args = vars(parser.parse_args())
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)