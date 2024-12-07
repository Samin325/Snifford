import torch
import torch.nn as nn
import torch.optim as optim
from model import IDSModel
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model = IDSModel()
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # convert data to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # training loop
    for epoch in range(epochs):
        model.train()

        # shuffle data and create mini-batches
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

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
