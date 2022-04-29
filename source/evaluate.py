import torch
from source.nseloss import NSELoss
from source.hyperpara import device


def evaluate_model(model, test_loader):
    model.eval()

    # Loss and optimizer
    criterion = NSELoss()

    with torch.no_grad():
        preds = []
        obs = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            preds.append(outputs)
            obs.append(targets)

        # concatenate in dim 0
        preds = torch.cat(preds)
        obs = torch.cat(obs)
        loss = criterion(preds, obs)

    model.train()
    return loss.item()
