# Snifford - AI-powered Intrusion Detection System

Snifford is an AI-powered intrusion detection system (IDS) designed to analyze network traffic and identify potential security threats, distinguishing between benign and malicious activity in real-time. Its importance lies in its ability to proactively defend against cyberattacks, safeguard sensitive data, and enhance the overall security posture of modern digital environments.

The data sets used so far for this project include the CIC-IDS2017 (availablle [here](https://www.unb.ca/cic/datasets/ids-2017.html)) and UNSW-NB15 (available [here](https://research.unsw.edu.au/projects/unsw-nb15-dataset)).

We prioritize recall as a metric to evaluate our model to help ensure that all malicious traffic is identified, but raw accuracy is still valued to ensure the false positive rate stays low. The LSTM model that we built achieved 94.86% accuarcy on the hold out test-set, but it successfully detected 98.12% of all malicious activity. The confusion matrix for the model is as follows (the trained model can be found at this [link](https://drive.google.com/file/d/1zBrTbLao3Wu5nbB-B-qFfb-mqsWEhKcg/view?usp=drive_link)):

Confusion Matrix:
|                       | Predicted Benign   | Predicted Malicious  |
| --------------------- | ------------------ | -------------------- |
| Actually Benign       | 427659             | 27000                |
| Actually Malicious    | 2099               | 109394               |

The following is a rudimentary example of Sniffords ability to detect malicious traffic - an alert is generated when it flags something as malicious and then all the extracted features of the network traffic are printed to the terminal:
![image](https://github.com/user-attachments/assets/639d98bc-aebc-489d-b7dd-3a957ba0a63b)

This will be cleaned up to produce human-readable alerts on a graphical user interface
