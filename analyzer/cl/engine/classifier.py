import torch
import torch.nn.functional as F

def knn_classifier(model, test_data_loader, k_knn=200, t_knn=0.1):
    '''kNN classifier as a monitor of progress by computing accuracy.'''
    model.eval()

    with torch.no_grad():
        pass


def knn_predict(feat_v, feat_set, classes, k_knn, t_knn):
    '''Predict the class of a single feature vector by computing the
        k nearest neighbours within a feat_set.
        Args:
            - feat_v: (torch.Tensor) single feature vector.
            - feat_set: (torch Tensor) Fixed set of various feature vectors.
            - classes:
            - k_knn: (int) Hyperparameter of knn algorithm -- among k nearest.
    '''
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=k_knn, dim=-1)

    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / t_knn).exp()

    one_hot_label = torch.zeros(feature.size(0) * k_knn, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores.argsort(dim=-1, descending=True)
