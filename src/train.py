import torch
import torch.nn as nn
import torch.optim as optim
from src.SniffNet import IDSModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm 

def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model = IDSModel()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # convert data to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # training loop
    for epoch in range(epochs):
        model.train()

        epoch_loss = 0  # Initialize loss accumulator
        batch_progress = tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}/{epochs}', unit='batch')

        # shuffle data and create mini-batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            # ensure target labels are of shape (batch_size, 1)
            batch_y = batch_y.unsqueeze(1)  # add a dimension to match output shape

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_progress.set_postfix(loss=epoch_loss/(i+1))

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    # evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        _, predicted = torch.max(output, 1)
        accuracy = accuracy_score(y_test, predicted)
        print(f'Accuracy on test set: {accuracy * 100:.2f}%')

    # save the model
    torch.save(model.state_dict(), 'results/trained_model.pth')

    # return labels and predictions for visualization
    return y_test, predicted
