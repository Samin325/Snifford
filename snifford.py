from src.data_loader import load_and_preprocess_data, split_data
from src.train import train_model
from src.utils import plot_confusion_matrix

data_folder = 'data/csv/MachineLearningCVE'

# load and preprocess data
data = load_and_preprocess_data(dataset_folder)
X_train, X_test, y_train, y_test = split_data(data)

# train and evaluate model
train_model(X_train, y_train, X_test, y_test)

# visualize  confusion matrix
classes = ['Benign', 'Malicious'] 
plot_confusion_matrix(y_test, y_pred, classes)
