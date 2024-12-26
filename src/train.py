import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.SniffNet import IDSModel


def train_model(X_train, y_train, X_test, y_test, model_path, epochs=10, batch_size=32):
    model = IDSModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # convert data to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        # training loop
        for epoch in range(epochs):
            model.train()

            epoch_loss = 0
            total_samples = 0  # keeps track of actual number of data points processed to accurately calculate average
            batch_progress = tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch + 1}/{epochs}',
                                  unit='minibatch')

            # shuffle data and create minibatches
            for i in batch_progress:
                minibatch_X = X_train_tensor[i:i + batch_size]
                minibatch_y = y_train_tensor[i:i + batch_size].unsqueeze(
                    1)  # ensure target labels are of shape (batch_size, 1)

                # backpropagation
                optimizer.zero_grad()
                output = model(minibatch_X)
                loss = criterion(output, minibatch_y)
                loss.backward()
                optimizer.step()

                # calculate epoch loss and update progress bar
                epoch_loss += loss.item() * len(minibatch_X)
                total_samples += len(minibatch_X)
                batch_progress.set_postfix(loss=epoch_loss / total_samples)
                batch_progress.update(batch_size)

            batch_progress.close()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / total_samples:.4f}')

        # save the model
        torch.save(model.state_dict(), model_path)

    # evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        predicted = (torch.sigmoid(output) > 0.5).float()
        y_pred = predicted.cpu().numpy()  # move to CPU and convert to NumPy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy on test set: {accuracy * 100:.2f}%')

    return y_test, y_pred
