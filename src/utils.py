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


def plot_confusion_matrix_terminal(y_true, y_pred, classes=None):
    if classes:
        class_labels = classes
    else:
        class_labels = ["Benign", "Malicious"]

    # get confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # if matrix is 2x2, print out formatted
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        print("\nConfusion Matrix:")
        print(f"{'':<20}{'Predicted Benign':<20}{'Predicted Malicious':<20}")
        print(f"{'Actually Benign':<20}{tn:<20}{fp:<20}")
        print(f"{'Actually Malicious':<20}{fn:<20}{tp:<20}")

        print("\nDetailed Breakdown:")
        print(f"True Negatives (Benign correctly classified as Benign): {tn}")
        print(f"False Positives (Benign misclassified as Malicious): {fp}")
        print(f"False Negatives (Malicious misclassified as Benign): {fn}")
        print(f"True Positives (Malicious correctly classified as Malicious): {tp}")
    else:
        print("\nConfusion Matrix (raw):")
        print(cm)
