import torch
import torch.nn as nn
from source.evaluate import evaluate_model
from source.hyperpara import learning_rate, device, print_every, num_epochs


def train_model(model, train_loader, test_loader=None):
    model.train()
    total_step = len(train_loader)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # mini-batch gradient descend
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            # inputs = inputs.reshape(-1, seq_length, input_size).to(device)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (test_loader is not None) and ((i + 1) % print_every == 0):
                val_loss = evaluate_model(model, test_loader)
                print(f'Epoch [{epoch + 1}/{num_epochs}],',
                      f'Step [{i + 1}/{total_step}],',
                      f'Loss: {loss.item():.4f},',
                      f'Val Loss: {val_loss:.4f}')
