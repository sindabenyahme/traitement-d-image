#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:04:52 2021

@author: chaari
"""

import numpy as np
from numpy import linalg as LA
from PIL import Image
from scipy.io import *


def pMRI_simulator(S, ref, sigma, R):
    Nc = S.shape[2]
    Size = S.shape[0]
    Size_red = round(Size/R)
    delta = round(Size_red/2)
    reduced_FoV = np.zeros((Size_red, Size, Nc))
    for j in range(Nc):
        for m in range(Size_red):
            for n in range(Size):
                indices = []
                for r in range(0, R):
                    indices.append((m+delta+r*Size_red) % Size)
                s = S[indices, n, :].transpose()
                A_des = ref[indices, n]
                noise = np.random.normal(0, sigma, Nc)
                A_obs = np.dot(s, A_des) + noise
                reduced_FoV[m, n, :] = A_obs
    return reduced_FoV




def reconstruct(reduced_FoV, S, psi, Lambda):
    [Size_red, Size, Nc] = reduced_FoV.shape
    delta = round(Size_red/2)
    reconstructed = np.zeros((Size, Size))
    psi_1 = np.linalg.pinv(psi)
    R = round(Size/Size_red)
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


import numpy as np

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


def SignalToNoiseRatio(x_reference, x):
    # Convertir les images en tableaux NumPy
    ref_array = np.array(x_reference, dtype=np.float64)
    reconstructed_array = np.array(x, dtype=np.float64)
    # Calculer le signal moyen
    signal_mean = np.mean(reconstructed_array)
    # Calculer l'écart type du bruit
    noise_std = np.std(reconstructed_array - ref_array)
    snr = 20*np.log10(signal_mean / noise_std)

    return abs(snr)


# Question 4
#
