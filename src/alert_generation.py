import torch
from src.SniffNet import LSTMModel
from src.data_loader import load_and_preprocess_data

def process_traffic(samples, model):
    samples_tensor = torch.tensor(samples, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(samples_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # convert logits to binary predictions

    # generate alerts
    for sample, prediction in zip(samples, predictions):
        if prediction == 1:
            print("Alert: malicious activity detected:")
            print(f"Sample Details: {sample}")

if __name__ == "__main__":

    model_path = "results/LSTMModel_verion_1.pth" 
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data_folder = 'data/csv/MachineLearningCVE'
    processed_data_path = 'data/ids_dataset1.pkl'

    traffic_samples = load_and_preprocess_data(data_folder, processed_data_path)

    # process the samples and generate alerts
    process_traffic(traffic_samples, model)
