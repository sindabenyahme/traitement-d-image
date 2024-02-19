#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:28:50 2021

@author: chaari
"""
#from mat4py import loadmat
#import h5py
import scipy.io
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import log






# Chargement des données
import scipy.io
import matplotlib.pyplot as plt

# Chargement des données
import scipy.io
import matplotlib.pyplot as plt

# Chargement des données
loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']

# Paramètres de simulation
sigma_values = [5, 10, 15]  # Différentes intensités de bruit à tester
R = 2

# Simulation des données avec différentes intensités de bruit
for sigma in sigma_values:
    reduced_FoV = pMRI_simulator(S, ref, sigma, R)
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
    for j in range(min(8, reduced_FoV.shape[2])):  # Pour chaque antenne (maximum 8)
        ax = axes[j // 4, j % 4]  # Calcul des indices pour les sous-graphiques
        ax.imshow(reduced_FoV[:, :, j],)  # Affichage de l'antenne en niveaux de gris
        ax.set_title(f'Antenne {j+1}')
        ax.axis('off')
    
    plt.suptitle(f"Images simulées avec un bruit sigma={sigma}", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Évaluation de l'impact du bruit sur la qualité des images
    reconstructed = reconstruct(reduced_FoV, S, sigma * np.eye(S.shape[2]), Lambda=1)
    snr = SignalToNoiseRatio(ref, reconstructed)
    print(f"SNR for sigma={sigma}: {snr} dB")




import scipy.io
import matplotlib.pyplot as plt

# Chargement des données
loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']

sigma_values = [5, 10, 15] 
R_values = [2, 4]  # Valeurs de R à comparer

for R in R_values:
    print(f"Comparaison pour R={R}")
    for sigma in sigma_values:
        reduced_FoV = pMRI_simulator(S, ref, sigma, R)
        
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
        for j in range(min(8, reduced_FoV.shape[2])):
            ax = axes[j // 4, j % 4]  
            ax.imshow(reduced_FoV[:, :, j])  
            ax.set_title(f'Antenne {j+1}')
            ax.axis('off')

        plt.suptitle(f"Images simulées avec R={R} et un bruit sigma={sigma}", fontsize=16)
        plt.tight_layout()
        plt.show()

        
        reconstructed = reconstruct(reduced_FoV, S, sigma * np.eye(S.shape[2]), Lambda=1)
        snr = SignalToNoiseRatio(ref, reconstructed)
        print(f"SNR for sigma={sigma}: {snr} dB")

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Chargement des données
loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']

# Paramètres de simulation
sigma_values = [5, 10, 15]  # Différentes intensités de bruit à tester
R_values = [2, 4]  # Valeurs de R à comparer

# Fonction de reconstruction
def reconstruct(reduced_FoV, S, psi, Lambda, R):
    [Size_red, Size, Nc] = reduced_FoV.shape
    delta = round(Size_red/2)
    reconstructed = np.zeros((Size, Size))
    psi_1 = np.linalg.pinv(psi)
    for m in range(Size_red):
        for n in range(Size):
            indices = []
            for r in range(0, R):
                indices.append((m+delta+r*Size_red) % Size)
            s = S[indices, n, :].transpose()
            A = reduced_FoV[m, n, :]
            reconstructed[indices, n] = np.dot(np.dot(np.linalg.pinv(np.dot(
                np.dot(A, psi_1), A.transpose()) + Lambda*np.eye(Nc)), np.dot(A, psi_1)), s)
    return reconstructed

# Simulation des données avec différentes valeurs de R et de bruit
for R in R_values:
    print(f"Comparaison pour R={R}")
    for sigma in sigma_values:
        reduced_FoV = pMRI_simulator(S, ref, sigma, R)
        
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
        for j in range(min(8, reduced_FoV.shape[2])):  # Pour chaque antenne (maximum 8)
            ax = axes[j // 4, j % 4]  # Calcul des indices pour les sous-graphiques
            ax.imshow(reduced_FoV[:, :, j], cmap='gray')  # Affichage de l'antenne en niveaux de gris
            ax.set_title(f'Antenne {j+1}')
            ax.axis('off')

        plt.suptitle(f"Images simulées avec R={R} et un bruit sigma={sigma}", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Reconstruction de l'image à partir des données simulées
        reconstructed = reconstruct(reduced_FoV, S, sigma * np.eye(S.shape[2]), Lambda=1, R=R)

        # Affichage de l'image reconstruite
        plt.imshow(reconstructed, cmap='gray')
        plt.title("Image Reconstruite")
        plt.axis('off')
        plt.show()

        # Calcul du SNR pour évaluer la qualité de l'image reconstruite
        snr = SignalToNoiseRatio(ref, reconstructed)
        print(f"SNR for R={R}, sigma={sigma}: {snr} dB")
        
        
        
        

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Chargement des données
loaded = scipy.io.loadmat('reference.mat')
ref = loaded['im']
loaded = scipy.io.loadmat('sens.mat')
S = loaded['s']

# Paramètres de simulation
sigma_values = [5, 10, 15]  # Différentes intensités de bruit à tester
R_values = [2, 4]  # Valeurs de R à comparer

# Fonction de reconstruction avec régularisation de Tikhonov
def reconstruct_tikhonov(reduced_FoV, S, psi, Lambda, R):
    [Size_red, Size, Nc] = reduced_FoV.shape
    delta = round(Size_red/2)
    reconstructed = np.zeros((Size, Size))
    psi_1 = np.linalg.pinv(psi)
    for m in range(Size_red):
        for n in range(Size):
            indices = []
            for r in range(0, R):
                indices.append((m+delta+r*Size_red) % Size)
            s = S[indices, n, :].transpose()
            A = reduced_FoV[m, n, :]
            
            # Vérifier si la matrice est de plein rang
            if np.linalg.matrix_rank(np.dot(A, psi_1)) == min(Nc, Size_red):
                regularization_term = Lambda * np.eye(Nc)
            else:
                regularization_term = Lambda * np.eye(Nc)  # Utiliser une régularisation supplémentaire si la matrice n'est pas de plein rang
            
            # Calcul de la pseudo-inverse avec régularisation de Tikhonov
            reconstructed[indices, n] = np.dot(np.dot(np.linalg.pinv(np.dot(
                np.dot(A, psi_1), A.transpose()) + regularization_term), np.dot(A, psi_1)), s)
    return reconstructed


# Simulation des données avec différentes valeurs de R et de bruit
for R in R_values:
    print(f"Comparaison pour R={R}")
    for sigma in sigma_values:
        reduced_FoV = pMRI_simulator(S, ref, sigma, R)
        
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
        for j in range(min(8, reduced_FoV.shape[2])):  # Pour chaque antenne (maximum 8)
            ax = axes[j // 4, j % 4]  # Calcul des indices pour les sous-graphiques
            ax.imshow(reduced_FoV[:, :, j], cmap='gray')  # Affichage de l'antenne en niveaux de gris
            ax.set_title(f'Antenne {j+1}')
            ax.axis('off')

        plt.suptitle(f"Images simulées avec R={R} et un bruit sigma={sigma}", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Reconstruction de l'image à partir des données simulées avec régularisation de Tikhonov
        reconstructed_tikhonov = reconstruct_tikhonov(reduced_FoV, S, sigma * np.eye(S.shape[2]), Lambda=1, R=R)

        # Affichage de l'image reconstruite avec régularisation de Tikhonov
        plt.imshow(reconstructed_tikhonov, cmap='gray')
        plt.title("Image Reconstruite avec régularisation de Tikhonov")
        plt.axis('off')
        plt.show()

        # Calcul du SNR pour évaluer la qualité de l'image reconstruite avec régularisation de Tikhonov
        snr_tikhonov = SignalToNoiseRatio(ref, reconstructed_tikhonov)
        print(f"SNR for R={R}, sigma={sigma} with Tikhonov regularization: {snr_tikhonov} dB")

fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(reduced_FoV[:,:,0])

fig.add_subplot(rows, columns, 2)
plt.imshow(reduced_FoV[:,:,1])

fig.add_subplot(rows, columns, 3)
plt.imshow(reduced_FoV[:,:,2])

fig.add_subplot(rows, columns, 4)
plt.imshow(reduced_FoV[:,:,3])

fig.add_subplot(rows, columns, 5)
plt.imshow(reduced_FoV[:,:,4])

fig.add_subplot(rows, columns, 6)
plt.imshow(reduced_FoV[:,:,5])

fig.add_subplot(rows, columns, 7)
plt.imshow(reduced_FoV[:,:,6])

fig.add_subplot(rows, columns, 8)
plt.imshow(reduced_FoV[:,:,7])

snr= SignalToNoiseRatio(ref, reconstructed)
print(snr)


plt.imshow(ref)
plt.show()


