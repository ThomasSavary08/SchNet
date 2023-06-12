### Importation des bibliothèques ###
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import utils
import os


### Distribution des énergies à prédire ###
train_energies = pd.read_csv('data/energies/train.csv')
plt.figure(figsize = (10,6))
sns.histplot(train_energies["energy"], kde = True, stat = "count", binwidth = 4)
plt.title("Distribution de l'énergie dans le jeu d'entraînement")
plt.savefig('Images/train_energies_dist.png')

### Compte du nombres d'atomes dans les différentes molécules ###
liste_training_file = os.listdir('data/atoms/train')
liste_nb_atoms_train = []
liste_atoms_train = []
for file in liste_training_file:
    lines = open('data/atoms/train/' + file) .read().split('\n')
    nb_atoms = int(lines[0])
    liste_nb_atoms_train.append(nb_atoms)
    for i in range(2, 2 + nb_atoms):
        liste_atoms_train.append(lines[i].split()[0])

liste_test_file = os.listdir('data/atoms/test')
liste_nb_atoms_test = []
liste_atoms_test = []
for file in liste_test_file:
    lines = open('data/atoms/test/' + file) .read().split('\n')
    nb_atoms = int(lines[0])
    liste_nb_atoms_test.append(nb_atoms)
    for i in range(2, 2 + nb_atoms):
        liste_atoms_test.append(lines[i].split()[0])

### Nombre d'atomes par molécule ###
fig, axes = plt.subplots(1, 2, figsize = (15, 5))
sns.histplot(liste_nb_atoms_train, kde = True, discrete = True, binrange = (4,24), ax = axes[0])
axes[0].set_title("Train")
axes[0].set_xlabel("Nombre d'atomes")
axes[0].set_ylabel("Nombre de molécules")
sns.histplot(liste_nb_atoms_test, kde = True, discrete = True, binrange = (4,24), ax = axes[1])
axes[1].set_title("Test")
axes[1].set_xlabel("Nombre d'atomes")
axes[1].set_ylabel("Nombre de molécules")
plt.savefig('Images/nb_atoms_mol.png')

### Quels atomes sont présents dans les données ###
fig, axes = plt.subplots(1, 2, figsize = (15, 5))
sns.histplot(liste_atoms_train, discrete = True, stat = "percent", ax = axes[0])
axes[0].set_title("Train")
axes[0].set_xlabel("Atomes")
axes[0].set_ylabel("%")
sns.histplot(liste_atoms_test, discrete = True, stat = "percent", ax = axes[1])
axes[1].set_title("Test")
axes[1].set_xlabel("Atomes")
axes[1].set_ylabel("%")
plt.savefig('Images/nb_atoms.png')

### Affichage 3D d'une molécule ###
mol = utils.read_xyz('data/atoms/train/' + liste_training_file[13], "numpy", False, False)
Z, x, y, z = mol[:,0], mol[:,1], mol[:,2], mol[:,3]
Zs = {6: 'C', 1: 'H', 8: 'O', 7: 'N', 16: 'S', 17: 'Cl'}
dict = {'Z': [Zs[i] for i in Z], 'x': x, 'y': y, 'z': z}
mol_df = pd.DataFrame(dict)
print(mol_df)
fig = px.scatter_3d(mol_df, x = 'x', y = 'y', z = 'z', color = 'Z', title = r"$C_{4}H_{4}O_{2}$")
fig.show()