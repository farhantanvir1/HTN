#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 16:57:11 2021

@author: illusionist
"""
# model_hetero.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_scatter import scatter_add


class HTN(torch.nn.Module):

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False, predictor_hidden_dim=64):
        super().__init__()


        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        self.htn_layers = nn.ModuleList()

        for i in range(num_of_layers):
            layer = HTNLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            self.htn_layers.append(layer)

        """

        FT 042923 self.predictor is a sequential neural network consisting of two linear layers separated by a 
        ReLU activation function.
        
        The first layer takes the concatenated features of the drug, target, and disease nodes and projects them
        to a lower-dimensional space (predictor_hidden_dim). 
        
        The second layer takes the output of the first layer and predicts a single interaction value for the 
        given input. The ReLU activation function introduces non-linearity into the model, which helps it learn 
        complex patterns in the data.


        """

        # Add predictor network
        self.predictor = nn.Sequential(
            # First linear layer: takes the concatenated features of drug, target, and disease
            # as input (3 * num_features_per_layer[-1]) and outputs predictor_hidden_dim features
            nn.Linear(3 * num_features_per_layer[-1], predictor_hidden_dim),

            # Apply a ReLU activation function to introduce non-linearity
            nn.ReLU(),

            # Second linear layer: takes the output of the previous layer (predictor_hidden_dim features)
            # and outputs a single value, which represents the predicted interaction value
            nn.Linear(predictor_hidden_dim, 1)
        )

    def forward(self, in_nodes_features, edge_index):
        x = in_nodes_features
        for layer in self.htn_layers:
            x = layer(x, edge_index)
        return x

    def forward_predictor(self, out_nodes_features, drug_indices_batch, target_indices_batch, disease_indices_batch):
        z_drug = out_nodes_features[drug_indices_batch]
        z_target = out_nodes_features[target_indices_batch]
        z_disease = out_nodes_features[disease_indices_batch]

        # Combine drug, target, and disease features
        interaction_features = torch.cat((z_drug, z_target, z_disease), dim=-1)

        # Apply the predictor network to obtain interaction values
        interaction_values = self.predictor(interaction_features)
        return interaction_values



class HTNLayer(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_heads, attention_mlp_hidden=64, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        """

        FT 042923 This is a linear transformation defined for initial node features. it is used to 
        transform the input node features in the following manner:

        The number of input features (num_in_features) is provided as the first argument.
        The number of output features is set to num_of_heads * num_out_features. This is because, 
        for each head in the multi-head attention mechanism, the model learns a separate set of features.

        This transforms the input node features into a higher-dimensional space suitable for the 
        multi-head attention mechanism. 

        """
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.attention_mlp_hidden = attention_mlp_hidden

    
        # Define the attention MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.num_out_features * 3, self.attention_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.attention_mlp_hidden, 1),
        )

        # Initialize theta (used for combining input features and weighted sum of neighbor features)
        self.theta = nn.Parameter(torch.Tensor(1, self.num_of_heads, self.num_out_features))
        nn.init.xavier_uniform_(self.theta)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        
    
    """

    FT 042923 This method is very specific to our model. I wrote this part.

    The compute_attention_scores method is responsible for calculating the attention scores 
    for each triplet of nodes (i, j, k) in the graph. The method does the following:

    1. Extracts the projected features for nodes i, j, and k using the edge_index tensor.
    
    2. Concatenates the features of nodes i, j, and k into a single tensor.
    
    3. Feeds the concatenated features to the attention MLP (a small neural network) to compute 
    the raw attention scores.
    
    4. Applies a LeakyReLU activation function to the raw attention scores.
    
    The output is a tensor containing the attention scores for each triplet of nodes in the graph. 

    """


    def compute_attention_scores(self, in_nodes_features_proj, edge_index):
      
      # FT 042923 Get the projected features of node i, j, and k
      in_nodes_features_proj_i = in_nodes_features_proj[edge_index[0]]
      in_nodes_features_proj_j = in_nodes_features_proj[edge_index[1]]
      in_nodes_features_proj_k = in_nodes_features_proj[edge_index[2]]

      # FT 042923 Concatenates the features of nodes i, j, and k
      combinations = torch.cat((in_nodes_features_proj_i, in_nodes_features_proj_j, in_nodes_features_proj_k), dim=-1)
      
      # FT 042923 Compute the attention scores using the attention MLP
      attention_scores = self.attention_mlp(combinations).squeeze(-1)

      # FT 042923 Apply LeakyReLU activation
      attention_scores = self.leakyReLU(attention_scores)
      return attention_scores

    """

    FT 042923 This method is very specific to our model. I wrote this part.

    The aggregate_neighbors method is responsible for aggregating the features of neighbors for each node 
    in the graph using the attention scores computed in the previous step. The method performs the 
    following operations:

    1. Normalizes the attention scores using the Softmax function to obtain attention coefficients.
    
    2. Performs an element-wise multiplication of the projected features for nodes j and k to compute
    their feature products.

    3. Computes a weighted sum of the neighbor feature products for each node i using the attention 
    coefficients.

    4. Multiplies the input features by the learnable parameter theta and adds the weighted sum of 
    neighbor features to obtain the output features for each node.

    The output is a tensor containing the aggregated features for each node in the graph. 
    This tensor will be passed through further processing in the HTNLayer's forward method. 

    """


    def aggregate_neighbors(self, in_nodes_features_proj, edge_index, attention_scores):
      # FT 042923 Normalize the attention scores using Softmax
      attention_scores_softmax = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
      
      # FT 042923 Element-wise multiplication of neighbor features (j and k)
      neighbor_product = in_nodes_features_proj[edge_index[1]] * in_nodes_features_proj[edge_index[2]]
      
      # FT 042923 Aggregate neighbors for each node (i) using the attention coefficients
      weighted_sum = scatter_add(attention_scores_softmax * neighbor_product, edge_index[0], dim=0, dim_size=in_nodes_features_proj.size(0))

       # FT 042923 Combine the input features and the weighted sum of neighbor features using theta
      out_nodes_features = self.theta * in_nodes_features_proj + weighted_sum
      return out_nodes_features



    """

    FT 042923 This method combines and calls all other methods in HTNLayer class. 
    
    The forward method first linearly transforms the input node features using self.linear_proj. Then, 
    it computes the attention scores for each triplet of nodes and aggregates the neighbors for each node
    using the attention coefficients. Finally, it applies any specified activation function and returns 
    the transformed node features.

    """

    def forward(self, in_nodes_features, edge_index):

        # 1. Linearly transform node features (num_nodes, num_out_features)
        in_nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        
         # 2. Compute attention coefficients for each triplet of nodes.
        attention_scores = self.compute_attention_scores(in_nodes_features_proj, edge_index)

         # 3. Aggregate neighbors for each node using the attention coefficients.
        out_nodes_features = self.aggregate_neighbors(in_nodes_features_proj, edge_index, attention_scores)
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)
        if self.bias is not None:
            out_nodes_features += self.bias
        if self.activation is not None:
            out_nodes_features = self.activation(out_nodes_features)
        return out_nodes_features
