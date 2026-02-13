import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/gambar1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

k = np.sqrt(2)
sigmas = [1.0, 1.0*k, 1.0*(k**2)]

plt.figure(figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis('off')

for i, s in enumerate(sigmas):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=s)
    
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)

    plt.subplot(2, 4, i+2)
    plt.imshow(blurred, cmap='gray')
    plt.title(f"Blurred ($\sigma={s:.2f}$)")
    plt.axis('off')

    plt.subplot(2, 4, i+6)
    plt.imshow(mag, cmap='jet')
    plt.title(f"Gradient Mag ($\sigma={s:.2f}$)")
    plt.axis('off')

plt.tight_layout()
plt.savefig('01_space_feature_analysis.png')
plt.show()