import numpy as np
import cv2
import pywt
from scipy.linalg import svd
from scipy.signal import convolve2d
from math import sqrt

ALPHA = 4.11 # Scaling factor for embedding the watermark (controls intensity)

# Definizione dei casi con parametri specifici
def case1():
    return {
        "fibonacci_spiral": [(128, 128), (95, 98), (133, 191), (175, 66), (39, 143), (213, 182), (100, 21), (73, 234), (248, 85), (3, 77), (188, 257), (287, 163), (106, 301), (266, 12), (264, 263), (333, 101), (175, 339), (337, 224), (47, 353), (343, 15), (264, 339), (390, 149), (129, 399), (353, 298), (410, 51), (232, 409), (424, 219), (62, 437), (339, 376), (463, 111), (175, 466), (432, 300)],
        "block_size": 4,

    }

def case2():
    return {
        "fibonacci_spiral": [(384, 128), (351, 98), (389, 191), (431, 66), (295, 143), (469, 182), (356, 21), (329, 234), (504, 85), (259, 77), (444, 257), (249, 206), (362, 301), (198, 121), (255, 283), (210, 7), (431, 339), (169, 196), (303, 353), (145, 65), (179, 287), (385, 399), (107, 153), (230, 373), (488, 409), (107, 259), (318, 437), (64, 89), (151, 364), (431, 466), (46, 205), (237, 452)],
        "block_size": 4,
    }

def case3():
    return {
        "fibonacci_spiral": [(128, 384), (95, 354), (133, 447), (175, 322), (39, 399), (213, 438), (100, 277), (73, 490), (248, 341), (3, 333), (172, 241), (287, 419), (31, 246), (266, 268), (119, 187), (333, 357), (238, 192), (337, 480), (38, 168), (343, 271), (171, 132), (390, 405), (312, 181), (77, 103), (410, 307), (242, 103), (424, 475), (388, 199), (140, 55), (463, 367), (323, 101), (19, 48)],
        "block_size": 4,
    }

def case4():
    return {
        "fibonacci_spiral": [(384, 384), (351, 354), (389, 447), (431, 322), (295, 399), (469, 438), (356, 277), (329, 490), (504, 341), (259, 333), (428, 241), (249, 462), (287, 246), (198, 377), (375, 187), (210, 263), (494, 192), (169, 452), (294, 168), (145, 321), (427, 132), (203, 188), (107, 409), (333, 103), (120, 249), (498, 103), (226, 114), (64, 345), (396, 55), (123, 169), (46, 461), (275, 48)],
        "block_size": 4,
    }

def case5():
    return {
        "fibonacci_spiral": [(256, 256), (223, 226), (261, 319), (303, 194), (167, 271), (341, 310), (228, 149), (201, 362), (376, 213), (131, 205), (316, 385), (300, 113), (121, 334), (415, 291), (159, 118), (234, 429), (394, 140), (70, 249), (392, 391), (247, 59), (127, 411), (461, 229), (82, 135), (303, 467), (366, 64), (41, 324), (465, 352), (166, 40), (175, 481), (471, 143), (17, 193), (392, 467)],
        "block_size": 4,
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

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

# Estrazione parametri utilizzati per l'embedding del watermark, ritorna la combinazione spirale-block_size corretta
def extract_embedding_params(original_image, watermarked_image, possible_configurations):
    best_combination = None
    min_diff = float('inf')

    # Itera sulle combinazioni di centro e block_size
    for config in possible_configurations:
        # Genera la spirale di Fibonacci a partire dal centro attuale
        fibonacci_spiral = config["fibonacci_spiral"]
        block_size = config["block_size"]
        total_diff = 0
        valid_blocks = 0

        # Itera sui blocchi della spirale
        for i, (x, y) in enumerate(fibonacci_spiral):
            # Controlla se il blocco è dentro i confini dell'immagine
            if x + block_size > original_image.shape[1] or y + block_size > original_image.shape[0]:
                continue

            # Estrazione dei blocchi dall'immagine originale e watermarked
            original_block = original_image[y:y + block_size, x:x + block_size]
            watermarked_block = watermarked_image[y:y + block_size, x:x + block_size]

            # Trasformata wavelet di Haar sui blocchi
            Coeff_orig = pywt.wavedec2(original_block, wavelet='haar', level=1)
            Coeff_wm = pywt.wavedec2(watermarked_block, wavelet='haar', level=1)

            # Estrai la sotto-banda LL
            LL_orig = Coeff_orig[0]
            LL_wm = Coeff_wm[0]

            # Decomposizione SVD della sotto-banda LL
            U_orig, S_orig, V_orig = np.linalg.svd(LL_orig)
            U_wm, S_wm, V_wm = np.linalg.svd(LL_wm)

            # Calcola la differenza tra i valori singolari
            diff = np.linalg.norm(S_wm - S_orig)
            total_diff += diff
            valid_blocks += 1

        # Calcola la differenza media per questa combinazione
        if valid_blocks > 0:
            avg_diff = total_diff / valid_blocks

            # Trova la combinazione che minimizza la differenza
            if avg_diff < min_diff:
                min_diff = avg_diff
                best_combination = (fibonacci_spiral, block_size)

    return best_combination

# Estrazione watermark da un'immagine data
def extract_watermark(original_image, watermarked_image, fibonacci_spiral, block_size, alpha=ALPHA):
    # Inizializzazione dell'array che conterrà il watermark estratto
    extracted_watermark = np.zeros(1024)

    for i, (x, y) in enumerate(fibonacci_spiral):
        # Estrazione del blocco originale e watermarked
        original_block = original_image[x:x + block_size, y:y + block_size]
        watermarked_block = watermarked_image[x:x + block_size, y:y + block_size]

        # Applicazione della trasformata wavelet sui blocchi
        Coeff_orig = pywt.wavedec2(original_block, wavelet='haar', level=1)
        Coeff_wm = pywt.wavedec2(watermarked_block, wavelet='haar', level=1)

        # Estrazione della sotto-banda LL (low-low)
        LL_orig = Coeff_orig[0]
        LL_wm = Coeff_wm[0]

        # Decomposizione SVD sui blocchi LL
        U_orig, S_orig, V_orig = np.linalg.svd(LL_orig)
        U_wm, S_wm, V_wm = np.linalg.svd(LL_wm)

        # Estrazione dei valori singolari del watermark
        Sw_extracted = (S_wm - S_orig) / alpha

        # Inserimento del watermark estratto nella matrice del watermark
        # Usando la stessa logica per selezionare gli indici come nell'embedding
        idx = (i * LL_orig.shape[0]) % 32

        # Aggiunta dei valori singolari estratti al watermark finale
        extracted_watermark[idx:idx + Sw_extracted.shape[0]] = Sw_extracted[:min(block_size, 32 - idx)]
    return extracted_watermark

# Funzione di rilevamento del watermark
def detection(input1, input2, input3):
        
    # Leggere tutte le immagini
    original_image = cv2.imread(input1, 0)
    watermarked_image = cv2.imread(input2, 0)
    attacked_image = cv2.imread(input3, 0)

    # Estrazione dei parametri utilizzati per l'embedding
    params = extract_embedding_params(original_image, watermarked_image, [ case1(),case2(),case3() ])

    # Variabile per memorizzare la similitudine più bassa e il watermark estratto
    similarity_W_A = float('inf')

    # Soglia τ per la similitudine
    threshold_tau = 0.75  # Esempio di soglia, da calibrare

    # Estrazione del watermark dall'immagine watermarked e da quella attaccata
    Wextracted = extract_watermark(original_image, watermarked_image, params[0], params[1])
    Wattacked = extract_watermark(original_image, attacked_image, params[0], params[1])

    # Similitudine tra i watermark
    similarity_W_A = similarity(Wextracted, Wattacked)
        
    # Calcolare il wPSNR tra l'immagine watermarked e quella attaccata
    wpsnr_value = wpsnr(watermarked_image, attacked_image)

    # Se la similitudine è inferiore alla soglia τ e il wPSNR è superiore a 35, l'attacco è riuscito
    if similarity_W_A < threshold_tau:
        output1 = 0  # Watermark distrutto
    else:
        output1 = 1  # Watermark presente
    output2 = wpsnr_value  # Restituisce il miglior wPSNR calcolato

    return output1, output2
