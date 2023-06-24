### Importation des bibliothèques ###
import matplotlib.pyplot as plt
import numpy as np
import torch
import model
import utils

### Définition des paramètres ###
gamma, mu_min, mu_max, nb_points = 10., 0., 5., 201
fts_dim, dim_inter = 64, 32 
nb_inter = 5
maxZ = 18

### Instanciation du réseau ###
net = model.BouzidNet(maxZ, gamma, mu_min, mu_max, nb_points, fts_dim, nb_inter, dim_inter)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

### Dataset d'entraînement ###
energies_path = "data/energies"
atoms_path = "data/atoms"
data = utils.TrainingDataset(energies_path, atoms_path, utils.read_xyz)
training_set, val_set = torch.utils.data.random_split(data, [0.8,0.2])
n_train, n_val = training_set.__len__(), val_set.__len__()

### Data Loader ###
train_dataloader = torch.utils.data.DataLoader(training_set, batch_size = 128, shuffle = True, num_workers = 4)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = n_val, shuffle = False, num_workers = 4)

### Nombre d'epochs et score pour l'enregistrement du modèle ###
n_epochs = 1000
best_score = 0.5

### Loss, optimizer et scheduler ###
criterion = torch.nn.MSELoss(reduction = "mean")
optimizer = torch.optim.AdamW(net.parameters(), lr = 1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, 1e-6, verbose = True)
list_val_loss = []

### Training ###
print("Début de l'entraînement...")
print("")

### Boucle sur le nombre d'epochs ###
for epoch in range(1, n_epochs + 1):

	print("Epoch n°" + str(epoch))

	### Boucle sur les batchs ###
	for X_train, y_train in train_dataloader:
        	
		X_train, y_train = X_train.to(device), y_train.to(device)
		outputs = net.forward(X_train)
		loss = criterion(outputs, y_train)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	### Evaluation sur le jeu de validation ###
	with torch.no_grad():
		for X_val, y_val in val_dataloader:
			outputs = net.forward(X_val)
			loss_val = torch.sqrt(criterion(outputs,y_val))
			list_val_loss.append(loss_val.item())
			print("Validation loss: " + str(loss_val.item()))
			print("")
    
	### MAJ du learning rate ###
	scheduler.step()
	
	### Enregistrement du modèle ###
	if (loss_val.item() < best_score):
		best_score = loss_val.item()
		torch.save(net, 'BouzidNet.pt')

### Affichage de l'évolution de la loss ###
plt.figure()
plt.title("Evolution de la loss de validation au cours de l'entraînement")
plt.plot(np.asarray(list_val_loss))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('Images/loss_evolution.png')
