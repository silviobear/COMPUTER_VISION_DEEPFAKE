import torch
import numpy as np
import scipy.fftpack
import cv2

def compute_2d_dct(image_tensor):
    # Passaggio in scala di grigi
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    if img_np.shape[2] == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np.squeeze()
        
    # Calcolo DCT
    dct_coeffs = scipy.fftpack.dct(
        scipy.fftpack.dct(img_gray, axis=0, norm='ortho'), 
        axis=1, norm='ortho'
    )
    return dct_coeffs