from semilearn.algorithms import SSLAlgorithm
from semilearn.utils.data import SSLDataLoader
import torch
from tqdm import tqdm

def ssl_train(model, algorithm:SSLAlgorithm, optimizer, train_loader, nepochs, device="cpu"):


    model.to(device)
    model.train()

    training_bar = tqdm(range(nepochs), desc="Training", leave=True)
    for epoch in training_bar:

        total_loss = 0.0
        total_batches = 0

        for i, batch in enumerate(train_loader, start=0):

            batch.to(device)

            optimizer.zero_grad()

            loss = algorithm.train_step(model, batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        training_bar.set_postfix(avg_loss = round(avg_loss, 4))


def ssl_train_cyclic(model, algorithm:SSLAlgorithm, optimizer, train_loader, num_iters, num_log_iters = 8, device="cpu"):


    model.to(device)
    model.train()

    training_bar = tqdm(train_loader, total=num_iters, desc="Training", leave=True)
    total_loss = 0.0

    for i, batch in enumerate(training_bar):


        batch.to(device)

        optimizer.zero_grad()

        loss = algorithm.train_step(model, batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


        avg_loss = total_loss / (i+1)

        if i % num_log_iters == 0:
            training_bar.set_postfix(avg_loss = round(avg_loss, 4))

        if i > num_iters:
            break
