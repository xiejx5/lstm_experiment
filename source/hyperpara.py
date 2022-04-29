import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
k_fold = 10
seq_length = 30
hidden_size = 64
output_size = 1
num_layers = 1
drop_prob = 0.2
batch_size = 64
num_epochs = 50
learning_rate = 0.0001
print_every = 10
