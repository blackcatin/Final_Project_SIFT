import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./images/gambar1.jpg')
img2 = cv2.imread('./images/gambar2.jpg')
img3 = cv2.imread('./images/gambar3.jpg')

sift = cv2.SIFT_create()

def get_robust_matches(im_a, im_b):
    """Fungsi untuk mendeteksi, mencocokkan, dan memfilter inliers antar dua citra."""
    kp_a, des_a = sift.detectAndCompute(im_a, None)
    kp_b, des_b = sift.detectAndCompute(im_b, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) > 10:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        inlier_count = np.sum(mask)
        precision = (inlier_count / len(good)) * 100
        draw_params = dict(matchColor=(0, 255, 0), 
                           matchesMask=mask.ravel().tolist(), 
                           flags=2)
        res_img = cv2.drawMatches(im_a, kp_a, im_b, kp_b, good, None, **draw_params)
        return res_img, inlier_count, precision
    return None, 0, 0

res12, count12, prec12 = get_robust_matches(img1, img2)
res23, count23, prec23 = get_robust_matches(img2, img3)
plt.subplot(2, 1, 1)
if res12 is not None:
    plt.imshow(cv2.cvtColor(res12, cv2.COLOR_BGR2RGB))
    plt.title(f"Pair 1 (Left-Center): {count12} Inliers ({prec12:.1f}% Precision)")
plt.axis('off')

plt.subplot(2, 1, 2)
if res23 is not None:
    plt.imshow(cv2.cvtColor(res23, cv2.COLOR_BGR2RGB))
    plt.title(f"Pair 2 (Center-Right): {count23} Inliers ({prec23:.1f}% Precision)")
plt.axis('off')

plt.tight_layout()
plt.savefig('05_robust_matching_3img.png')