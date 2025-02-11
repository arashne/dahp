import torch


def pairwise_l2_distance(x: torch.Tensor):
    n = x.shape[0]

    dist2 = torch.zeros((n, n), device=x.device)

    for i in range(n):
        for j in range(n):
            dist2[i, j] = ((x[i] - x[j])**2).mean()

    return torch.sqrt(dist2)


def pairwise_cosine_similarity(x: torch.Tensor):
    x = x / x.norm(dim=1, keepdim=True)

    cosine_sim = torch.matmul(x, x.T)

    return cosine_sim
