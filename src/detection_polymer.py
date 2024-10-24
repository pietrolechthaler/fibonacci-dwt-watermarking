import numpy as np
import cv2
import pywt
from scipy.linalg import svd
from scipy.signal import convolve2d
from math import sqrt

# Definizione dei casi con parametri specifici
def case1():
    return {
        "block_size": 4,
        "alpha": 4.11,
        "fibonacci_spiral": [
            (384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), 
            (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), 
            (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), 
            (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)
        ]
    }

def case2():
    return {
        "block_size": 4,
        "alpha": 4.11,
        "fibonacci_spiral": [
            (384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), 
            (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), 
            (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), 
            (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)
        ]
    }

def case3():
    return {
        "block_size": 4,
        "alpha": 4.11,
        "fibonacci_spiral": [
            (384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), 
            (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), 
            (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), 
            (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)
        ]
    }

# Dizionario switch
switch = {
    1: case1,
    2: case2,
    3: case3
}

# Funzione WPSNR
def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    difference = img1 - img2
    if not np.any(difference):
        return float('inf')  # Somiglianza perfetta
    csf = np.array([[0.98, 1.02, 0.98], [1.02, 1.06, 1.02], [0.98, 1.02, 0.98]])  # Esempio di CSF
    ew = convolve2d(difference, np.rot90(csf, 2), mode='valid')
    decibels = 20.0 * np.log10(1.0 / sqrt(np.mean(np.mean(ew ** 2))))
    return decibels

# Funzione di rilevamento del watermark
def detection(input1, input2, input3):
    # Leggere tutte le immagini
    original_image = cv2.imread(input1, 0)
    watermarked_image = cv2.imread(input2, 0)
    attacked_image = cv2.imread(input3, 0)

    # Estrazione watermark da un'immagine data
    def extract_watermark(image, fibonacci_spiral, block_size, alpha):
        watermark_extracted = np.zeros((32, 32))
        for i, (x, y) in enumerate(fibonacci_spiral):
            if x + block_size > image.shape[1] or y + block_size > image.shape[0]:
                continue

            block = image[x:x + block_size, y:y + block_size]
            Coefficients = pywt.wavedec2(block, wavelet='haar', level=1)
            LL = Coefficients[0]
            U, S, V = svd(LL)

            # Estrazione del valore singolare come watermark
            watermark_extracted[i // 32, i % 32] = S[0]

        return watermark_extracted
    
    # Variabile per memorizzare la similitudine più bassa e il watermark estratto
    best_similarity = float('inf')
    best_wpsnr = 0

    # Soglia τ per la similitudine
    threshold_tau = 0.1  # Esempio di soglia, da calibrare

    # Definizione dei parametri di embedding utilizzati
    cases = [1,2,3]
    for case in cases:
        params = switch.get(case)()

        # Estrazione del watermark dall'immagine watermarked e da quella attaccata
        Wextracted = extract_watermark(watermarked_image, params["fibonacci_spiral"], params["block_size"], params["alpha"])
        Wattacked = extract_watermark(attacked_image, params["fibonacci_spiral"], params["block_size"], params["alpha"])

        # Similitudine tra i watermark (usando una semplice metrica di differenza)
        similarity = np.linalg.norm(Wextracted - Wattacked)
        
        # Calcolare il wPSNR tra l'immagine watermarked e quella attaccata
        wpsnr_value = wpsnr(watermarked_image, attacked_image)


        # Se la similitudine è inferiore alla soglia τ e il wPSNR è superiore a 35, l'attacco è riuscito
        if best_similarity < threshold_tau and best_wpsnr >= 35:
            output1 = 0  # Watermark distrutto
        else:
            output1 = 1  # Watermark presente

        output2 = best_wpsnr  # Restituisce il miglior wPSNR calcolato

    return output1, output2
