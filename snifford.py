import numpy as np

from src.data_loader import load_and_preprocess_data, split_data
from src.train import train_model
from src.utils import plot_confusion_matrix_terminal

data_folder = 'data/csv/MachineLearningCVE'
processed_data_path = 'data/ids_dataset1.pkl'
model_path = 'results/initial_model.pth'

# load and preprocess data
data = load_and_preprocess_data(data_folder, processed_data_path)
X_train, X_test, y_train, y_test = split_data(data)

# train and evaluate model
y_test, y_pred = train_model(X_train, y_train, X_test, y_test, model_path)
y_pred = np.array(y_pred).flatten()

# visualize  confusion matrix
classes = ['Benign', 'Malicious']
plot_confusion_matrix_terminal(y_test, y_pred, classes)
