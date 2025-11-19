# fl_simulation.py
# Minimal PyTorch-based components for FL simulation:
# - Simple CNN
# - synthetic client data generator
# - local_train (adds optional gaussian DP noise to params)
# - aggregate_weights (FedAvg)
# - evaluate

import copy
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def synthetic_client_data(num_samples=200, input_shape=(1,28,28), num_classes=10, seed=None):
    """
    Returns (x, y) torch tensors for synthetic data.
    Use seed to create slightly different distributions per client.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, num_classes, (num_samples,))
    return x, y

def local_train(model, data, epochs=1, lr=0.01, dp_noise_std=0.0, device='cpu'):
    """
    Train a copy of 'model' on given data and return the trained state_dict.
    If dp_noise_std > 0, add gaussian noise to parameters to simulate local DP.
    """
    model = copy.deepcopy(model).to(device)
    model.train()
    x, y = data
    x = x.to(device)
    y = y.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    batch_size = 32

    for e in range(epochs):
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = x[idx]
            yb = y[idx]
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()

    # Simulate DP by adding Gaussian noise to parameters (for prototype/demo only)
    if dp_noise_std and dp_noise_std > 0:
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.normal(0, dp_noise_std, size=p.shape, device=p.device)
                p.add_(noise)

    return model.state_dict()

def aggregate_weights(state_dicts):
    """
    Simple FedAvg: average state_dict tensors elementwise (equal weighting).
    """
    agg = {}
    n = len(state_dicts)
    for k in state_dicts[0].keys():
        agg[k] = sum([sd[k].float() for sd in state_dicts]) / n
    return agg

def evaluate(model_state, model_constructor, data, device='cpu'):
    """
    Load model_state into a fresh model from model_constructor() and evaluate on data.
    Returns accuracy (0..1)
    """
    model = model_constructor().to(device)
    model.load_state_dict(model_state)
    model.eval()
    x, y = data
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        out = model(x)
        preds = out.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

if __name__ == '__main__':
    # Quick smoke test: simulate 3 clients, 3 rounds
    device = 'cpu'
    global_model = SimpleCNN()
    clients = 3
    client_data = [synthetic_client_data(seed=i) for i in range(clients)]
    rounds = 3
    for r in range(rounds):
        sds = []
        for i in range(clients):
            sd = local_train(global_model, client_data[i], epochs=1, lr=0.01, dp_noise_std=0.01, device=device)
            sds.append(sd)
        agg = aggregate_weights(sds)
        global_model.load_state_dict(agg)
        accs = [evaluate(global_model.state_dict(), lambda: SimpleCNN(), d) for d in client_data]
        print(f"Round {r+1} client accs:", accs)
