import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def generate_rf_features(z, edges):
    features = []
    for u, v in edges:
        features.append(np.concatenate([z[u], z[v], [z[u] @ z[v]]]))
    return np.array(features)


class RFFusion:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.rf = RandomForestClassifier(n_estimators=100)

    def train_rf(self, z_val, val_edges, val_edges_false):
        # Generate features for edges for Random Forest
        val_edges_all = np.concatenate([val_edges, val_edges_false])
        val_labels_all = np.concatenate([
            np.ones(len(val_edges)), np.zeros(len(val_edges_false))
        ])
        val_features = generate_rf_features(z_val, val_edges_all)
        self.rf.fit(val_features, val_labels_all)

    def evaluate_with_rf(self, z_val, z_test,
                         val_edges, val_edges_false,
                         test_edges, test_edges_false,
                         A_pred_test):

        # Evaluate with RF fusion
        self.train_rf(z_val.cpu().numpy(), val_edges, val_edges_false)

        test_edges_all = np.concatenate([test_edges, test_edges_false])
        test_labels_all = np.concatenate([
            np.ones(len(test_edges)), np.zeros(len(test_edges_false))
        ])
        test_features = generate_rf_features(z_test.cpu().numpy(), test_edges_all)

        rf_probs = self.rf.predict_proba(test_features)[:, 1]
        vg_gin_probs = A_pred_test.cpu().numpy()[test_edges_all[:, 0], test_edges_all[:, 1]]
        final_probs = self.alpha * vg_gin_probs + (1 - self.alpha) * rf_probs

        test_roc_auc = roc_auc_score(test_labels_all, final_probs)
        test_ap = average_precision_score(test_labels_all, final_probs)

        return test_roc_auc, test_ap