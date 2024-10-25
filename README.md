# Digital Image Watermarking Using Fibonacci Spiral and DWT-SVD

## Abstract
Digital image watermarking is a technique used to embed hidden information, or "watermarks," within an image to protect copyright, verify authenticity, and ensure content integrity. This project explores a robust watermarking approach that combines Fibonacci spiral positioning with the Discrete Wavelet Transform (DWT) and Singular Value Decomposition (SVD) for embedding. The Fibonacci spiral, a unique geometric structure, is used to determine key embedding locations within the image, ensuring spatial robustness and reducing visibility of the watermark. DWT-SVD is then applied to these selected regions, leveraging the transform's ability to localize frequency and spatial information, thus enhancing resistance against image processing attacks such as compression, blurring, noise, and resizing. Evaluation of the watermarked images is performed using Weighted Peak Signal-to-Noise Ratio (wPSNR), assessing watermark resilience and perceptual quality across multiple attack scenarios. This approach aims to provide an effective balance between robustness and image fidelity.

## Description
This watermarking algorithm try to be an innovative embedding strategy that combines a Fibonacci spiral for selecting key positions within the image with a combined approach of Discrete Wavelet Transform (DWT) and Singular Value Decomposition (SVD) for embedding the watermark. The primary goal is to ensure that the watermark is both resistant to attacks and visually imperceptible.

The watermarking process begins with the selection of embedding points, guided by five predefined Fibonacci spirals centered at various locations within the image. At each selected embedding point, the image undergoes a wavelet transform, which separates it into frequency bands. Embedding primarily occurs in the low-frequency (LL) band due to its resilience against common attacks such as compression and resizing. The DWT enables precise control over frequency localization, which aids in both robust embedding and reduced visibility of the watermark.

Once the LL band is isolated, SVD is applied to decompose this band into singular component matrices (U, S, V). The watermark is embedded by subtly modifying the singular values (S) based on a scaling factor (`ALPHA`), which controls the watermark intensity. This SVD-based modification ensures that the watermark is resilient to typical distortions without introducing visible artifacts.
The `ALPHA` parameter determines the watermark’s intensity, balancing resilience to attacks and visual transparency. By adjusting `ALPHA`, the algorithm can increase robustness (making the watermark harder to remove) while keeping the watermark minimally intrusive to the original image’s appearance.

The algorithm evaluates each Fibonacci spiral’s effectiveness by embedding the watermark with each spiral configuration and then subjecting the watermarked image to various simulated attacks, including Gaussian blur, JPEG compression, and noise addition. For each spiral, the `Weighted Peak Signal-to-Noise Ratio (wPSNR)` between the original and attacked images is calculated. The spiral that results in the highest average wPSNR across all attack types is chosen, as this indicates the optimal balance between robustness and visual quality.

## File Structure

```
📦 polymer/
├── 📁 src/
│   ├── 📄 embedding_polymer.py #watermark embedding script
│   ├── 📄 detection_polymer.py #watermark detection script
│   ├── 📄 attacks.py #attack watermark images script
│   ├── 📄 roc_polymer.py #roc generation script
│   ├── 📄 generation_watermark.py #script to generate random watermark
│   └──📁 utilities/
│       ├── 📄 csf.csv
│       └── 📄 watermark.npy #generated watermark file
├── 📁 data/ 
│   ├── 📄 sample_image1.bmp #sample grayscale images
│   ├── 📄 sample_image2.bmp
│   ├── 📄 ...
│   ├── 📄 sample_imageN.bmp

├── 📄 README.md
├── 📄 requirements.txt
└── 📄 LICENSE
```


## Environment Setup
To ensure compatibility and reproducibility, this project requires Python 3.8.10.

1. Install Python 3.8.10
Ensure you have Python 3.8.10 installed. You can download it from python.org if it’s not already installed.

2. Create a virtual environment using Python 3.8.10
`python3.8 -m venv .env`

3. Activate the virtual environment
`source .env/bin/activate`

4. Install Dependencies
With the virtual environment activated, install the required packages from requirements.txt
`pip install -r requirements.txt`

## Usage

## Generate Watermark
To create a binary watermark for embedding, you can use `generation_watermark.py` script to generate a random watermark and save it as a .npy file in `src/utilities` folder.
```bash
python generation_watermark.py
```

### Embedding 
To embed a watermark in an image, use the `embedding()` function, which integrates a watermark into the specified image using the Fibonacci spiral and DWT-SVD method. The function accepts the paths to both the original image and the watermark file (saved in .npy format) and returns the best watermarked image after evaluating its robustness under different attack scenarios.

```python
import embedding_polymer

original_image_path = 'path/to/original_image'
watermark_path = 'path/to/watermark'
watermarked = embedding_polymer.embedding(original_image_path, watermark_path)
```

### Detection
The `detection()` function evaluates the integrity of a watermarked image after an attack by comparing the watermarks extracted from both the original watermarked image and the attacked image. 
The function uses similarity and wPSNR values to determine whether the watermark has been significantly degraded, indicating a successful attack.

```python
import detection_polymer
import cv2

original_image_path = 'path/to/original_image'
watermarked_image_path = 'path/to/watermarked_image'

original_image = cv2.imread(original_image_path)
watermarked_image = cv2.imread(watermarked_image_path)

watermarked_image_path = 'path/to/watermarked/image'
watermark_extracted = detection_polymer.detection(original_image, watermarked_image, watermarked_image)
```

The detection function outputs:
1. The `attack success status`:
- 1 if the attack was successful, meaning the watermark has been significantly compromised.
- 0 if the attack was unsuccessful, meaning the watermark is still detectable.
2. The `wPSNR value` between the watermarked and attacked images.

### ROC curves
### Attacks

## Contributors

- Collizzolli Leonardo
- Graziadei Ylenia
- Lechthaler Pietro


