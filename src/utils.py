import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes=None):

    # map numeric labels to class names if provided
    if classes and set(y_true).issubset({0, 1}):
        labels = [0, 1]
        xticklabels = yticklabels = classes
    else:
        labels = sorted(set(y_true) | set(y_pred))
        xticklabels = yticklabels = labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()
