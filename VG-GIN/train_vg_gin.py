import random

import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time

import model_vg_gin
from input_data import load_data, extract_edge_index
from perturbation import add_feature_noise, add_edge_perturbation, device
from preprocessing import *
import args

import warnings

from rf_ensemble import RFFusion

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

adj, features = load_data(args.dataset)

# extract edge_index
edge_index = extract_edge_index(adj)
edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
if edge_index.dim() == 2 and edge_index.shape[0] != 2:
    edge_index = edge_index.transpose(0, 1)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train
# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Calculate weights for positive samples
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Add self-loops
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Transform PyTorch sparse tensor
adj_norm = torch.sparse_coo_tensor(torch.LongTensor(adj_norm[0].T),
                            torch.FloatTensor(adj_norm[1]), 
                            torch.Size(adj_norm[2])).to(device)
adj_label = torch.sparse_coo_tensor(torch.LongTensor(adj_label[0].T),
                            torch.FloatTensor(adj_label[1]), 
                            torch.Size(adj_label[2])).to(device)
features = torch.sparse_coo_tensor(torch.LongTensor(features[0].T),
                            torch.FloatTensor(features[1]), 
                            torch.Size(features[2])).to(device)

f = features.to_dense().to(device)

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0)).to(device)
weight_tensor[weight_mask] = pos_weight

def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []

    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data.cpu()))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    # Calculate ROC_AUC and AP
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

# Initialize lists
test_roc_scores = []
test_ap_scores = []
train_losses = []

for experiment in range(2):
    print(f"Starting experiment {experiment + 1}")

    # Early stopping parameters
    patience = 20
    best_val_roc = 0
    patience_counter = 0

    # Initialize model and optimizer
    model = model_vg_gin.GIN_VGAE(args.input_dim, args.gin_hidden_dim, args.output_dim, args.num_gin_layers).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Initialize RF fusion
    rf_fusion = RFFusion()

    for epoch in range(args.num_epoch):
        t = time.time()
        model.train()

        if random.random() < 0.5:
            perturbed_f = add_feature_noise(f, noise_scale=0.1)
        else:
            perturbed_f = f.clone()
        if random.random() < 0.3:
            perturbed_edge_index = add_edge_perturbation(
                edge_index,
                num_nodes=f.size(0),
                perturb_ratio=0.1
            )
        else:
            perturbed_edge_index = edge_index.clone()
        A_pred, mean, logstd, z = model(perturbed_f, perturbed_edge_index)

        # Calculate loss
        optimizer.zero_grad()
        loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1),
                                                       weight=weight_tensor)
        kl_divergence = -0.5 / A_pred.size(0) * (1 + 2 * logstd - mean ** 2 - torch.exp(logstd) ** 2).sum(1).mean()
        main_loss = loss + kl_divergence
        aux_loss, aux_details = model.compute_auxiliary_losses()
        total_loss = main_loss + aux_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            A_pred_val, _, _, z_val = model(f, edge_index)
            val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred_val)
            train_acc = get_acc(A_pred, adj_label)
        model.train()

        # Early stopping logic
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Record training loss
        train_losses.append(total_loss.item())

        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(total_loss.item()),
              "train_acc=", "{:.5f}".format(train_acc),
              "val_roc=", "{:.5f}".format(val_roc),
              "val_ap=", "{:.5f}".format(val_ap),
              "time=", "{:.5f}".format(time.time() - t))

    model.eval()
    with torch.no_grad():
        A_pred_test, _, _, z_test = model(f, edge_index)
        test_roc_auc, test_ap = rf_fusion.evaluate_with_rf(
            z_val, z_test,
            val_edges, val_edges_false,
            test_edges, test_edges_false,
            A_pred_test
        )

    test_roc_scores.append(test_roc_auc)
    test_ap_scores.append(test_ap)
    print("End of training!",
          "test_roc=", "{:.5f}".format(test_roc_auc),
          "test_ap=", "{:.5f}".format(test_ap))

# Calculate statistics
test_roc_mean = np.mean(test_roc_scores)
test_roc_std = np.std(test_roc_scores, ddof=1)
test_ap_mean = np.mean(test_ap_scores)
test_ap_std = np.std(test_ap_scores, ddof=1)

print("Test ROC_AUC: {:.3f} ± {:.3f}".format(test_roc_mean, test_roc_std))
print("Test AP: {:.3f} ± {:.3f}".format(test_ap_mean, test_ap_std))