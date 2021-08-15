from torchvision.datasets.utils import download_and_extract_archive
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch
import os
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from csq import CSQLightening
from util import get_labels_pred_closest_hash_center

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 100)

filename = 'TM.zip'
data_name = "TM"
data_dir = './data'

CHECKPT_PATH='/scratch/gobi1/rexma/scCSQ/checkpoints/TM/CSQ-epoch=09-Val_F1_score_median_CHC_epoch=0.667.ckpt'
N_CLASS=55
N_FEATURES=19791

with open(os.path.join("label_maps", data_name, "label_mapping.json")) as f:
            label_mapping = json.load(f)

model = CSQLightening.load_from_checkpoint(checkpoint_path=CHECKPT_PATH,                                                        n_class=N_CLASS,
                                           n_features=N_FEATURES)

if not os.path.exists(data_dir+ '/' + data_name):
    url = "https://github.com/Aprilhuu/Deep-Learning-in-Single-Cell-Analysis/raw/main/TM.zip"
    download_and_extract_archive(url, data_dir, filename=filename)    

print("Download succeeded!")

DataPath = data_dir + "/" + data_name + "/Filtered_TM_data.csv"
LabelsPath = data_dir + "/" + data_name + "/Labels.csv"

labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
data = pd.read_csv(DataPath, index_col=0, sep=',')

print("Data loaded!")

input_data = torch.from_numpy(data.values).float()
binary_predict = model.forward(input_data).sign()
labels_pred_CHC = get_labels_pred_closest_hash_center(binary_predict.detach().numpy(), labels.to_numpy(), model.hash_centers.numpy())
string_labels = [label_mapping[str(int_label)] for int_label in labels_pred_CHC]

print("Prediction done!")
print(binary_predict.shape)

tsne = TSNE()
X_embedded = tsne.fit_transform(binary_predict.detach().numpy())

sns_plot = sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=string_labels, legend='full')

plt.title("TM (scDeepHash)")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")

sns_plot.figure.savefig("./plots/TM_scDeepHash.png")

