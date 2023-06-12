### Importation des bibliothèques ###
import torch
import utils


### Bloc de convolution ###
class CFConv(torch.nn.Module):
    
    ### Initialisation d'un bloc de convolution
    ### INPUT
    ### gamma : Facteur d'échelle de la rbf
    ### mu_min : centre minimal de la rbf
    ### mu_max : centre maximal de la rbf
    ### nb_points : nombre de points entre mu_min et mu_max
    ### fts_dim : dimension des features
    ### act : fonction d'activation
    ### OUTPUT
    ### Module correspondant au bloc cfconv de l'article
    def __init__(self, gamma, mu_min, mu_max, nb_points, fts_dim):
        super().__init__()
        self.gamma_ = torch.tensor([-gamma])
        self.min_mu = mu_min
        self.max_mu = mu_max
        self.dim_lin = nb_points
        self.dim_fts = fts_dim
        self.act_ = utils.SSP
        self.dense1 = torch.nn.Linear(nb_points, fts_dim)
        self.dense2 = torch.nn.Linear(fts_dim, fts_dim)

    ### Passage de X et R à travers le bloc
    ### INPUT
    ### X : matrices des features pour les différents atomes
    ### R : matrices des coordonnées des différents atomes
    ### OUTPUT:
    ### new_X : sortie du bloc, voir l'article pour plus de détails
    def forward(self, X, R, Mask):

        ### Batch_size et nombre d'atoms ###
        bs, n = X.shape[0], X.shape[2]
                
        ### Matrice des distances entre les atomes
        dist = torch.cdist(R, R, p = 2)
        
        ### Vecteur des différentes valeurs de mu
        mu_vector = torch.linspace(self.min_mu, self.max_mu, self.dim_lin, dtype = torch.float)
        mu_vector = mu_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, n, n, self.dim_lin)
        
        ### Calcul de la rbf pour les différentes distances
        dist = dist.squeeze(1).unsqueeze(-1).expand(bs, n, n, self.dim_lin)
        rbf = torch.pow(torch.sub(mu_vector, dist), 2)
        rbf = torch.exp(torch.mul(self.gamma_, rbf))
        
        ### Passage à travers les couches denses et les activations
        W = self.dense1(rbf)
        W = self.act_(W)
        W = self.dense2(W)
        W = self.act_(W)

        ### Convolution ###
        output = torch.mul(W, X)
        output = torch.sum(output, dim = 2).unsqueeze(1)
        return torch.mul(output, Mask)


### Bloc d'interaction ###
class Interaction(torch.nn.Module):
    
    ### Initialisation d'un bloc d'interaction
    ### INPUT
    ### conv_bloc : bloc de convulution (présent dans chaque bloc d'interaction)
    ### fts_dim : dimension des features
    ### act : fonction d'activation
    ### OUTPUT
    ### Module correspondant au bloc Interaction de l'article
    def __init__(self, conv_bloc, fts_dim):
        super().__init__()
        self.cfconv_ = conv_bloc
        self.dim_fts = fts_dim
        self.act_ = utils.SSP
        self.ATW1 = torch.nn.Linear(self.dim_fts, self.dim_fts)
        self.ATW2 = torch.nn.Linear(self.dim_fts, self.dim_fts)
        self.ATW3 = torch.nn.Linear(self.dim_fts, self.dim_fts)

    ### Passage de X et R à travers le bloc
    ### INPUT
    ### X : matrices des features pour les différents atomes
    ### R : matrices des coordonnées des différents atomes
    ### OUTPUT:
    ### new_X : sortie du bloc, voir l'article pour plus de détails
    def forward(self, X, R, Mask):
        
        ### Passage dans le premier réseau atom-wise
        res = self.ATW1(X)
        res = torch.mul(X, Mask)
        
        ### Passage dans le bloc de convolution
        res = self.cfconv_(res, R, Mask)
        
        ### Passage les autres réseaux atom-wise + activation
        res = self.ATW2(res)
        res = torch.mul(res, Mask)
        res = self.act_(res)
        res = self.ATW3(res)
        res = torch.mul(res, Mask)
        
        ### Ajout du résidu à la sortie
        return torch.add(X, res)


### SchNet ###
class BouzidNet(torch.nn.Module):
    
    ### Initialisation d'un réseau SchNet
    ### INPUT
    ### maxZ : Nombre d'atomes différents présents dans les données
    ### gamma, mu_min, mu_max, nb_points, fts_dim, act_conv : paramètres des bloc de convolutions
    ### act_inter : fonction d'activation pour les blocs d'interaction
    ### nb_inter : nombre de bloc d'interactions
    ### dim_inter : dimension intermédiaire des features dans l'avant dernier réseau atom-wise
    ### act : fonction d'activation
    ### OUTPUT
    ### Module correspondant au réseau SchNet de l'article
    def __init__(self, maxZ, gamma, mu_min, mu_max, nb_points, fts_dim, nb_inter, dim_inter):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings = maxZ, embedding_dim = fts_dim, padding_idx = 0)
        self.interactions = torch.nn.ModuleList()
        for _ in range(nb_inter):
            conv_block = CFConv(gamma, mu_min, mu_max, nb_points, fts_dim)
            interaction_ = Interaction(conv_block, fts_dim)
            self.interactions.append(interaction_)
        self.atw_inter = torch.nn.Linear(fts_dim, dim_inter)
        self.atw_final = torch.nn.Linear(dim_inter, 1)
        self.act_ = utils.SSP
        self.fts_dim = fts_dim
        self.inter_dim = dim_inter
        
    ### Prédiction d'un énergie par le réseau
    ### INPUT
    ### network_input : tenseur d'entrée conmprenant l'indices des éléments et leurs positions (R)
    ### OUTPUT:
    ### predicted_energy : sortie du réseau correspondant à l'énergie prédite par le réseau
    def forward(self, network_input):
        
        ### Extraction des indices et de R ###
        Z = network_input[:,:,:,0].type(torch.int)
        R = network_input[:,:,:,1:].type(torch.float)
        
        ### Création du tenseur des features grâce à l'embedding ###
        X = self.embedding(Z)

        ### Mask ###
        mask_loss = Z.gt(0).type(torch.float).unsqueeze(-1)
        mask = X.ne(0.).type(torch.float)
        
        ### Passage de la matrices des features à travers les blocs d'interaction ###
        for interaction in self.interactions:
            X = interaction.forward(X, R, mask)
        
        ### Passage par les réseaux atom-wise + activation ###
        X = self.atw_inter(X)
        X = torch.mul(X, mask[:,:,:,:self.inter_dim])
        X = self.act_(X)
        X = self.atw_final(X)
        X = torch.mul(X, mask_loss)

        ### Calcul de l'énergie par Sum-pooling ###
        return torch.sum(X, axis = 2).squeeze(1)