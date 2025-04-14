from semilearn.algorithms import SSLAlgorithm
import torch
from tqdm import tqdm

def ssl_train(model, algorithm:SSLAlgorithm, optimizer, train_loader, nepochs, device="cpu"):


    model.to(device)
    model.train()

    training_bar = tqdm(range(nepochs), desc="Training", leave=True)
    for epoch in training_bar:
        # epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nepochs}")

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



            # epoch_bar.set_postfix(loss=loss.item(), iteration=i)
        avg_loss = total_loss / total_batches
        training_bar.set_postfix(avg_loss = round(avg_loss, 4))