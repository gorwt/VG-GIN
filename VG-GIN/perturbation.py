import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU ...")

def add_feature_noise(x, noise_scale=0.1):
    # Add Gaussian noise
    noise = torch.randn_like(x) * noise_scale
    return x + noise

def add_edge_perturbation(edge_index, num_nodes, perturb_ratio=0.1):
    num_edges = edge_index.size(1)

    # Randomly delete edges
    if perturb_ratio > 0:
        mask = torch.rand(num_edges) > perturb_ratio
        edge_index = edge_index[:, mask]

    # Randomly add edges
    num_new = int(num_edges * perturb_ratio)
    new_edges = torch.randint(0, num_nodes, (2, num_new)).to(device)
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    return edge_index