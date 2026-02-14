import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/gambar1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(img, None)

img_all_kp = cv2.drawKeypoints(img_rgb, kp, None, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_stable = sorted(kp, key=lambda x: x.response, reverse=True)[:50]
img_stable_kp = cv2.drawKeypoints(img_rgb, kp_stable, None, 
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(img_all_kp)
plt.title(f"All Detected Keypoints (Total: {len(kp)})")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_stable_kp)
plt.title("Top 50 Most Stable Keypoints (Filtered by Response)")
plt.axis('off')

plt.tight_layout()
plt.savefig('03_visualisasi_proses_keypoints.png')
plt.show()

print(f"Total keypoints yang ditemukan: {len(kp)}")