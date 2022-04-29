import pandas as pd
from source.lstm import LSTM
from source.datasplit import KFoldSplit
from source.datasets import HydroDataset
from source.train import train_model
from source.evaluate import evaluate_model
from source.hyperpara import (seq_length, hidden_size, k_fold, device,
                              drop_prob, output_size, num_layers)


def workflow(df):
    # Hydro dataset
    input_size = df.shape[1] - output_size
    dataset = HydroDataset(df, seq_length, input_size, output_size)

    # k fold evaluation
    n, accumulation = 0, 0
    for train_loader, test_loader in KFoldSplit(dataset):

        # define the network
        model = LSTM(input_size, hidden_size, output_size,
                     num_layers, drop_prob).to(device)

        # train the model
        train_model(model, train_loader, test_loader=None)

        # evaluate the model
        accumulation += evaluate_model(model, test_loader)
        n += 1

    # print k-fold score
    print(f'Input: {list(df.columns[:-1])}')
    print(f'{k_fold}-fold NSE: {accumulation / n}')

    return accumulation / n


# original input
df = pd.read_csv('data/02116500.csv')
score = workflow(df[['P', 'PET', 'Tmax', 'Tmin', 'R']])


# without Tmin
score = workflow(df[['P', 'PET', 'Tmax', 'R']])


# replace Tmin with Tmax-Tmin
df['Tdif'] = df['Tmax'] - df['Tmin']
score = workflow(df[['P', 'PET', 'Tmax', 'Tdif', 'R']])


# add humidity index as input
df['HI'] = df['P'] / df['PET']
score = workflow(df[['P', 'PET', 'HI', 'Tmax', 'Tmin', 'R']])
