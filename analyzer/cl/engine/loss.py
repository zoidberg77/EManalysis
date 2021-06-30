import torch.nn.functional as F

def similarity_func(p, z):
	'''negative cosine similarity for measuring the similarity of two vectors.'''
	return (- F.cosine_similarity(p, z.detach(), dim=-1).mean())
