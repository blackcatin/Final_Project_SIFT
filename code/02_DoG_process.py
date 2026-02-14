import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/gambar1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

s1, s2, s3 = 1.0, 1.6, 2.5
b1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s1)
b2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s2)
b3 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s3)

dog1 = cv2.subtract(b2, b1)
dog2 = cv2.subtract(b3, b2)
dog_norm = cv2.normalize(dog1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, thresh = cv2.threshold(dog_norm, 185, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis_blob = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    if radius > 2:
        cv2.circle(vis_blob, (int(x), int(y)), int(radius), (255, 0, 0), 2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(dog1, cmap='RdBu'); plt.title("DoG Level 1"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(dog2, cmap='RdBu'); plt.title("DoG Level 2"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(vis_blob); plt.title("Candidate Keypoints (Blobs)"); plt.axis('off')
plt.tight_layout()
plt.savefig('02_dog_blobs.png')
plt.show()