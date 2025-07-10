import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('MonaLisa.jpg').convert('L')  # Ganti 'sample_image.jpg' dengan nama file gambar Anda
image_array = np.array(image, dtype=float)

U, S, Vt = np.linalg.svd(image_array, full_matrices=False)

def recontruktion_image(U,S,Vt,k):
    return np.dot(U[:,:k]*S[:k],Vt[:k,:])

k_values = [10,50,100]

plt.figure(figsize=(15, 5))
plt.subplot(1, len(k_values) + 1, 1)
plt.title('Gambar Asli')
plt.imshow(image_array, cmap='gray')
plt.axis('off')

for i, k in enumerate(k_values):
    compressed_image = recontruktion_image(U, S, Vt, k)
    # Konversi kembali ke format gambar (uint8)
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    plt.subplot(1, len(k_values) + 1, i + 2)
    plt.title(f'k = {k}')
    plt.imshow(compressed_image, cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# original_size = image_array.size
# compressed_size = k_values[-1] * (U.shape[0] + Vt.shape[1]) + k_values[-1]
# compression_ratio = original_size / compressed_size
# print(f'Rasio kompresi untuk k={k_values[-1]}: {compression_ratio:.2f}x')