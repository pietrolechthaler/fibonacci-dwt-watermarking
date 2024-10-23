import numpy as np
import cv2
import pywt
from scipy.linalg import svd
from scipy.signal import convolve2d
from math import sqrt

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
    
    # Definizione dei parametri di embedding utilizzati
    block_size = 4
    n_blocks_to_embed = 32
    alpha = 4.11

    # Definire centri multipli (stessi centri dell'algoritmo di embedding)
    centers = [
        (original_image.shape[1] // 4, original_image.shape[0] // 4),
        (3 * original_image.shape[1] // 4, original_image.shape[0] // 4),
        (original_image.shape[1] // 4, 3 * original_image.shape[0] // 4),
        (3 * original_image.shape[1] // 4, 3 * original_image.shape[0] // 4),
        (original_image.shape[1] // 2, original_image.shape[0] // 2)
    ]

    # Generare la spirale di Fibonacci
    def generate_fibonacci_spiral(n, center, img_shape):
        max_radius = min(img_shape[0], img_shape[1]) / 2
        fibonacci_points = []
        phi = (1 + sqrt(5)) / 2  # Rapporto aureo

        i = 0
        while len(fibonacci_points) < n:
            theta = i * (2 * np.pi / phi)
            r = sqrt(i / n) * max_radius
            x = int(r * np.cos(theta)) + center[0]
            y = int(r * np.sin(theta)) + center[1]

            # Aggiungere solo punti entro i confini dell'immagine
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                fibonacci_points.append((x, y))

            i += 1

        return fibonacci_points

    # Estrazione watermark da un'immagine data
    def extract_watermark(image, fibonacci_spiral):
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

    for center in centers:
        # Generare la spirale per ogni centro
        fibonacci_spiral = generate_fibonacci_spiral(n_blocks_to_embed, center, original_image.shape)

        # Estrazione del watermark dall'immagine watermarked e da quella attaccata
        Wextracted = extract_watermark(watermarked_image, fibonacci_spiral)
        Wattacked = extract_watermark(attacked_image, fibonacci_spiral)

        # Similitudine tra i watermark (usando una semplice metrica di differenza)
        similarity = np.linalg.norm(Wextracted - Wattacked)

        # Calcolare il wPSNR tra l'immagine watermarked e quella attaccata
        wpsnr_value = wpsnr(watermarked_image, attacked_image)

        # Se questa è la migliore similitudine trovata, aggiornare i risultati
        if similarity < best_similarity:
            best_similarity = similarity
            best_wpsnr = wpsnr_value

    # Soglia τ per la similitudine
    threshold_tau = 0.1  # Esempio di soglia, da calibrare

    # Se la similitudine è inferiore alla soglia τ e il wPSNR è superiore a 35, l'attacco è riuscito
    if best_similarity < threshold_tau and best_wpsnr >= 35:
        output1 = 0  # Watermark distrutto
    else:
        output1 = 1  # Watermark presente

    output2 = best_wpsnr  # Restituisce il miglior wPSNR calcolato

    return output1, output2
