### Importation des bibliothèques ###
import pandas as pd
import numpy as np
import torch

### Rotation d'angle theta d'une matrice ###
def rotation(A, axis = "all", theta = np.pi):
    
    if (axis == "x"):
        Rx = np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]], np.float16)
        return A @ np.transpose(Rx)
    
    elif (axis == "y"):
        Ry = np.array([[np.cos(theta),0,np.sin(theta)], [0,1,0], [-np.sin(theta),0,np.cos(theta)]], np.float16)
        return A @ np.transpose(Ry)
    
    elif (axis == "z"):
        Rz = np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta),np.cos(theta),0], [0,0,1]], np.float16)
        return A @ np.transpose(Rz)
        
    else:
        Rx = np.array([[1,0,0], [0,np.cos(theta),-np.sin(theta)], [0,np.sin(theta),np.cos(theta)]], np.float16)
        Ry = np.array([[np.cos(theta),0,np.sin(theta)], [0,1,0], [-np.sin(theta),0,np.cos(theta)]], np.float16)
        Rz = np.array([[np.cos(theta),-np.sin(theta),0], [np.sin(theta),np.cos(theta),0], [0,0,1]], np.float16)
        return ((A @ np.transpose(Rx)) @ np.transpose(Ry)) @ np.transpose(Rz)


### Fonction permettant de convertir un ficher xyz en tenseur numpy ou torch ###
def read_xyz(file, output_type, padding, aug, nmax = 23):
    
    ### Ouverture du fichier ###
    xyz = open(file, "r")
    lignes = xyz.read().split('\n')
    
    ### Nombre d'atomes dans la molécule ###
    nb_atoms = int(lignes[0])
    
    ### Création d'un array numpy pour remplissage ###
    xyzmatrix = np.ndarray((nb_atoms, 4), dtype = "float")
        
    ### Dictionnaire de bijection entre le numéro atomique Z et l'atome ###
    Zs = {'C': 6, 'H': 1, 'O': 8, 'N': 7, 'S': 16, 'Cl': 17}
    
    ### Remplissage du tableau numpy ###
    i = 0
    for line in lignes[2:-1]:
        if len(line.split()) == 4:
            atom, x, y, z = line.split()
            xyzmatrix[i,:] = int(Zs[atom]), float(x), float(y), float(z)
            i += 1

    ### Augmentation ###
    if aug:
        do = np.random.rand(3)
        ### Permutation ###
        if (do[0] >= 0.5):
            xyzmatrix = np.random.permutation(xyzmatrix)
        ### Rotation ###
        if (do[1] >= 0.5):
            theta = 2*np.pi*np.random.rand(1)
            xyzmatrix[:,1:] = rotation(xyzmatrix[:,1:], axis = "all", theta = theta[0])
        ### Translation ###
        if (do[2] >= 0.5):
            u = np.random.rand(1,3)
            xyzmatrix[:,1:] += u 

    ### Padding ###
    if padding:
        to_add = np.zeros((nmax - xyzmatrix.shape[0], 4))
        xyzmatrix = np.concatenate((xyzmatrix,to_add), axis = 0)
    
    ### Type de la sortie ###
    if (output_type == "numpy"):
        return xyzmatrix
    else:
        return torch.from_numpy(xyzmatrix).unsqueeze(0)


### Création d'un dataset torch avec les données d'entraînement ###
class TrainingDataset(torch.utils.data.Dataset):
    
    ### Instanciation d'un dataset ###
    def __init__(self, energy_path, molecule_path, transformation):
        self.energy_path = energy_path
        self.mol_path = molecule_path
        self.transform_xyz = transformation
        self.label_ = pd.read_csv(self.energy_path + '/train.csv')
        
    
    ### Nombre d'éléments dans le dataset ###
    def __len__(self):
        return self.label_.shape[0]
    
    ### Obtenir un élément du dataset ###
    def __getitem__(self, idx):
        
        ### Lecture des informations dans le .csv ###
        id_, label = self.label_.iloc[idx,:]
        
        ### Lecture du fichier .xyz associé et conversion en tenseur ###
        file_path = self.mol_path + '/train/' + "id_" + str(int(id_)) + ".xyz"
        network_input = self.transform_xyz(file_path, "torch", True, False)
        
        ### Conversion de l'energy en tenseur ###
        label = torch.tensor([label], dtype = torch.float)

        return network_input,label


### Fonction d'activation Shifted-SoftPlus ###
def SSP(x):
    sfp = torch.nn.Softplus()
    return torch.sub(sfp(x),torch.log(torch.tensor([2.])))     
