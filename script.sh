# This script was used on a a Deep Learning VM on GCP with the PyTorch 2.3 (CUDA 12.1, Python 3.10) framework
# https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning

# set up virutal environment
python3 -m venv env
source env/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install obspy pandas scikit-learn
pip install -r requirements.txt 
pip install gdown

# download and extract data
mkdir results
cd results
gdown https://drive.google.com/file/d/1zBrTbLao3Wu5nbB-B-qFfb-mqsWEhKcg/view?usp=drive_link
cd ..
mkdir data
cd data
mkdir csv
cd csv
# wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip
# unzip GeneratedLabelledFlows.zip
# rm -rf GeneratedLabelledFlows.zip
wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip
unzip MachineLearningCSV.zip
rm -rf MachineLearningCSV.zip

# uncomment the following lines if wanting to train on pcap files as well

# cd ..
# mkdir pcap
# cd pcap
# wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs/Monday-WorkingHours.pcap
# wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap
# wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs/Wednesday-workingHours.pcap
# wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs/Thursday-WorkingHours.pcap
# wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap

cd ..
cd ..

# the following lines are a fix for an error faced: 
# https://discuss.pytorch.org/t/could-not-load-library-libcudnn-cnn-train-so-8-but-im-sure-that-i-have-set-the-right-ld-library-path/190277/2
cd /usr/local/cuda-12.1/lib64
sudo rm -f libcudnn*
cd /usr/local/cuda-12.1/include
sudo rm -f cudnn*
cd ~/Snifford/