### Importation des bibliothèques ###
import pandas as pd
import torch
import utils
import os

### Chargement du modèle ###
model = torch.load("BouzidNet.pt")
model.eval()

### Chargement des données ###
atoms_path = 'data/atoms/test'
files_list = os.listdir(atoms_path)

### Prédictions ###
list_id = []
list_pred = []

for file_name in files_list:
    id = file_name.split("_")[1].split(".")[0]
    mol_path = atoms_path + "/" + file_name
    x = utils.read_xyz(mol_path, "torch", True, False, nmax = 23).unsqueeze(0)
    pred = model.forward(x)
    list_id.append(id)
    list_pred.append(pred.item())

### Construction d'un dataset pandas ###
dict = {'id': list_id, 'energy': list_pred}
predictions_df = pd.DataFrame(dict)

### Sort par id ###
predictions_df['id'].astype(int)
predictions_df = predictions_df.sort_values('id')
predictions_df['id'] = predictions_df['id'].astype(str)

### Conversion en csv et enregistrement ###
submission_path = "submission.csv"
predictions_df.to_csv(submission_path, index = False)