### Importation des bibliothèques ###
import numpy as np
import torch
import utils

### Chargement du modèle ###
model = torch.load("BouzidNet.pt")
model.eval()

### Chargement d'une molécule pour l'exemple ###
mol_example = utils.read_xyz("data/atoms/train/id_1.xyz", "numpy", padding = False, aug = False)

### Prédiction sur la molécule originale ###
x_init = torch.from_numpy(mol_example).unsqueeze(0).unsqueeze(0)
y_init = np.around(model(x_init).item(), decimals = 2)

### Rotation de la molécule ###
theta = 2 * np.pi * np.random.rand()
mol_rotation = np.copy(mol_example)
mol_rotation[:,1:] = utils.rotation(mol_rotation[:,1:], "x", theta = theta)
x_rotation = torch.from_numpy(mol_rotation).unsqueeze(0).unsqueeze(0)
y_rotation = np.around(model(x_rotation).item(), decimals = 2)

### Translation de la molécule ###
u = np.random.rand(1,3)
mol_translation = np.copy(mol_example)
mol_translation[:,1:] += u
x_translation = torch.from_numpy(mol_translation).unsqueeze(0).unsqueeze(0)
y_translation = np.around(model(x_translation).item(), decimals = 2)

### Permutation de la molécule ###
mol_permutation = np.copy(mol_example)
mol_permutation = np.random.permutation(mol_permutation)
x_permutation = torch.from_numpy(mol_permutation).unsqueeze(0).unsqueeze(0)
y_permutation = np.around(model(x_permutation).item(), decimals = 2)

### Affichage des résultats ###
print("Énergie en sortie du réseau pour la molécule initiale: " + str(y_init))
print("Énergie en sortie du réseau pour la molécule permutée: " + str(y_permutation))
print("Énergie en sortie du réseau pour la molécule translatée: " + str(y_translation))
print("Énergie en sortie du réseau pour la molécule après rotation: " + str(y_rotation))