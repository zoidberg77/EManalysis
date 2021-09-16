import torch
import torch.nn.functional as F

def knn_classifier(model, feat_data_loader, test_data_loader, device, classes, k_knn=200, t_knn=0.1):
    '''kNN classifier as a monitor of progress by computing accuracy.'''
    model.eval()
    model.to(device)
    total_true, total_num, feat_set, gt_labels_set = 0.0, 0.0, list(), list()

    with torch.no_grad():
        for idx, (sample, _, gt_labels) in enumerate(feat_data_loader):
            features = model.forward(sample.to(device, non_blocking=True))
            feat_set.append(features.squeeze())
            gt_labels_set.append(gt_labels)
            if not idx % 100:
                print('iteration step: [{}/{}]'.format(idx, len(feat_data_loader)))

        feat_set = torch.cat(feat_set, dim=0).t().contiguous()
        gt_labels_set = torch.cat(gt_labels_set, dim=0).contiguous()

        for idx, (sample, _, gt_labels) in enumerate(test_data_loader):
            feature = model.forward(sample.to(device, non_blocking=True))
            pred_labels = knn_predict(feature, feat_set, gt_labels_set, classes, k_knn, t_knn)
            total_num += sample.size(0)
            total_true += (pred_labels[:, 0] == gt_labels).float().sum().item()

    accuracy = total_true / total_num
    return accuracy


def knn_predict(feat_v, feat_set, gt_labels_set, classes, k_knn, t_knn):
    '''Predict the class of a single feature vector by computing the
        k nearest neighbours within a feat_set.
        Args:
            - feat_v: (torch.Tensor) single feature vector.
            - feat_set: (torch Tensor) Fixed set of various feature vectors.
            - classes:
            - k_knn: (int) Hyperparameter of knn algorithm -- among k nearest.
    '''
    print('feature: {}, shape: {}'.format(type(feat_v), feat_v.shape))
    print('feature set: {}, shape: {}'.format(type(feat_set), feat_set.shape))
    sim_matrix = torch.mm(feat_v, feat_set)
    sim_weight, sim_indices = sim_matrix.topk(k=k_knn, dim=-1)

    sim_labels = torch.gather(gt_labels_set.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / t_knn).exp()

    one_hot_label = torch.zeros(feature.size(0) * k_knn, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    return pred_scores.argsort(dim=-1, descending=True)
