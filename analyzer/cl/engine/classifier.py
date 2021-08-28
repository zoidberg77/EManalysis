import torch
import torch.nn.functional as F

def knn_classifier(model, test_data_loader, k_knn=200, t_knn=0.1):
    '''kNN classifier as a monitor of progress by computing accuracy.'''
    model.eval()

    with torch.no_grad():
        pass
