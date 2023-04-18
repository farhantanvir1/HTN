#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:49:05 2021

@author: illusionist
"""

# util.py

import datetime
import errno
import numpy as np
import os
import pickle
import random
import torch
import scipy.sparse as sp
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from torch_geometric.data import download_url, extract_zip
from csv import DictReader
from csv import reader
import numpy as np
import pandas as pd
import xlrd
import csv
import os
import csv
import xlrd
import _thread
import time
import torch as th
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader

def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0] = 10000000000
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def negsamp_incr(drug_id_list, target_id_list, disease_id_list, drug_target_disease_list, n_drug_items=532, n_target_items=837, n_disease_items=270, n_samp=7238):
    """ Guess and check with arbitrary positivity check
    """
    neg_inds = []
    neg_drug_id=[]
    neg_target_id=[]
    neg_disease_id=[]
    while len(neg_inds) < n_samp:
        drug_samp = np.random.randint(0, n_drug_items)
        target_samp = np.random.randint(0, n_target_items)
        disease_samp = np.random.randint(0, n_disease_items)
        if [drug_samp, target_samp, disease_samp] not in drug_target_disease_list and [drug_samp, target_samp, disease_samp] not in neg_inds:
          neg_drug_id.append(drug_samp)
          neg_target_id.append(target_samp)
          neg_disease_id.append(disease_samp)
          neg_inds.append([drug_samp, target_samp, disease_samp])
    return neg_drug_id, neg_target_id, neg_disease_id, neg_inds

# The configuration below is from the paper.
default_configure = {
        "num_of_layers": 2,  # GNNs, contrary to CNNs, are often shallow (it ultimately depends on the graph properties)
        "num_heads_per_layer": [8, 1],
        "num_features_per_layer": [32, 8, 2],
        "add_skip_connection": False,  # hurts perf on Cora
        "bias": True,  # result is not so sensitive to bias
        "dropout": 0.6,  # result is sensitive to dropout
        "lr":0.005,
        "weight_decay": 0.001,
        "num_of_epochs": 100,
        "patience": 1000,
        'batch_size': 32
    }


def setup(args):
    args.update(default_configure)
    args['hetero']=True
    args['dataset'] = 'DrugBank' if args['hetero'] else 'DrugBank'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args
          
def negsamp_incr(drug_id_list, target_id_list, disease_id_list, drug_target_disease_list, n_drug_items=532, n_target_items=837, n_disease_items=270, n_samp=7238):
    """ Guess and check with arbitrary positivity check
    """
    neg_inds = []
    neg_drug_id=[]
    neg_target_id=[]
    neg_disease_id=[]
    while len(neg_inds) < n_samp:
        drug_samp = np.random.randint(0, n_drug_items)
        target_samp = np.random.randint(0, n_target_items)
        disease_samp = np.random.randint(0, n_disease_items)
        if [drug_samp, target_samp, disease_samp] not in drug_target_disease_list and [drug_samp, target_samp, disease_samp] not in neg_inds:
          neg_drug_id.append(drug_samp)
          neg_target_id.append(target_samp)
          neg_disease_id.append(disease_samp)
          neg_inds.append([drug_samp, target_samp, disease_samp])
    return neg_drug_id, neg_target_id, neg_disease_id, neg_inds
            
def load_drug_data(device):

    drug_path = '/content/drive/MyDrive/HTN/data/drug_feature.csv'
    target_path = '/content/drive/MyDrive/HTN/data/target_feature.csv'

    drug_df = pd.read_csv(drug_path)
    target_df = pd.read_csv(target_path)

    # Load the drug-target interaction dataset
    drug_target_interactions = pd.read_csv("/content/drive/MyDrive/HTN/data/drug_target_data_drugbank_final.csv")

    # Load the drug-disease interaction dataset
    drug_disease_interactions = pd.read_csv("/content/drive/MyDrive/HTN/data/drug_indication_data_drugbank_final.csv")

    drug_target_interactions_column_names = list(drug_target_interactions.columns)

    drug_disease_column_names = list(drug_disease_interactions.columns)

    # Create a dictionary of drugs
    unique_drug_id = drug_target_interactions["ID"].unique()
    unique_drug_id = pd.DataFrame(data={
        'drugId': unique_drug_id,
        'mappedID': pd.RangeIndex(len(unique_drug_id)),
    })

    # Create a dictionary of targets
    unique_target_id = drug_target_interactions["Protein"].unique()
    unique_target_id = pd.DataFrame(data={
        'targetId': unique_target_id,
        'mappedID': pd.RangeIndex(len(unique_target_id)),
    })


    # Create a dictionary of diseases
    unique_disease_id = drug_disease_interactions["indication_id"].unique()
    unique_disease_id = pd.DataFrame(data={
        'diseaseId': unique_disease_id,
        'mappedID': pd.RangeIndex(len(unique_disease_id)),
    })

    drug_target_disease_interactions = pd.merge(drug_target_interactions, drug_disease_interactions, how='inner', left_on = 'ID', right_on = 'Drug id')
    drug_target_disease_interactions = drug_target_disease_interactions.drop("Drug id",axis=1)
    
    drug_id = pd.merge(drug_target_disease_interactions['ID'], unique_drug_id,
                            left_on='ID', right_on='drugId', how='left')
    drug_id = torch.from_numpy(drug_id['mappedID'].values)

    target_id = pd.merge(drug_target_disease_interactions['Protein'], unique_target_id,
                                left_on='Protein', right_on='targetId', how='left')
    target_id = torch.from_numpy(target_id['mappedID'].values)

    disease_id = pd.merge(drug_target_disease_interactions['indication_id'], unique_disease_id,
                                left_on='indication_id', right_on='diseaseId', how='left')
    disease_id = torch.from_numpy(disease_id['mappedID'].values)


    # Concatenate both edge_index matrices
    edge_index = torch.column_stack([drug_id, target_id, disease_id])
    edge_index=edge_index.to(device)

    labels_np = edge_index.numpy()
    if np.isnan(labels_np).any() or np.isinf(labels_np).any():
      print("Warning: Labels contain NaN or infinite values.")

    drug_feature = pd.merge(unique_drug_id['drugId'], drug_df,
                            left_on='drugId', right_on='ID', how='left')
    drug_feature = drug_feature.drop(['ID','SMILES'], axis=1)
    drug_feature.rename( columns={'Unnamed: 0':'mappedID'}, inplace=True )

    drug_feat = np.zeros([532,32],dtype=int)
    for i in drug_feature.index:
      row = drug_feature.loc[i,'mappedID']
      for j in range(32):
        drug_feat[row][j] = drug_feature.loc[i,'EPFS_SMILES'][j+1]
    
    drug_features = torch.tensor(drug_feat, dtype=torch.float)

    target_feature = pd.merge(unique_target_id['targetId'], target_df,
                                left_on='targetId', right_on='ID', how='left')
    target_feature = target_feature.drop(['ID','ACS'], axis=1)
    target_feature.rename( columns={'Unnamed: 0':'mappedID'}, inplace=True )

    target_feat = np.zeros([836,32],dtype=int)
    for i in drug_feature.index:
      row = target_feature.loc[i,'mappedID']
      for j in range(32):
        target_feat[row][j] = target_feature.loc[i,'EPFS_ACS'][j+1]
    target_features = torch.tensor(target_feat, dtype=torch.float)

    disease_feat = np.random.randint(2, size=(270, 32))
    disease_features = torch.tensor(disease_feat, dtype=torch.float)
    # Concatenate node features
    all_node_features = torch.cat([drug_features, target_features, disease_features], dim=0)
    
    #Normalize the features (helps with training)
    all_node_features = normalize_features(all_node_features)
    all_node_features = torch.from_numpy(all_node_features).to(device)

    # Create the heterogeneous graph
    hetero_graph = Data(x=all_node_features, edge_index=edge_index).to(device)
    hetero_graph = hetero_graph.to(device)

    input_data_np = all_node_features.numpy()
    if np.isnan(input_data_np).any() or np.isinf(input_data_np).any():
      print("Warning: Input data contains NaN or infinite values.")

    
    #print('Datatype:', node_features_csr.dtype)

    #node_features_csr = torch.from_numpy(node_features_csr).float().to(device)

    drug_id_list = drug_id.tolist()

    target_id_list = target_id.tolist()

    disease_id_list = disease_id.tolist()

    drug_target_disease_list =[]

    for i in range(7238):
      lst=[]
      lst.append(drug_id_list[i])
      lst.append(target_id_list[i])
      lst.append(disease_id_list[i])
      drug_target_disease_list.append(lst)

    neg_drug_id, neg_target_id, neg_disease_id, neg_drug_target_disease_list = negsamp_incr(drug_id_list, target_id_list, disease_id_list, drug_target_disease_list)

    
    #graph_data = ()


    return drug_target_disease_list, neg_drug_target_disease_list, all_node_features, hetero_graph, drug_id, target_id, disease_id


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))