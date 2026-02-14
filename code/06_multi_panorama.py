import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_homography(img1, img2):
    """Fungsi pembantu untuk mendapatkan matriks Homography antara dua citra."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good) > 10:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        return H, kp1, kp2, good
    return None, None, None, None

def stitch_images(img1, img2):
    """Menggabungkan dua citra (img1 akan di-warp ke koordinat img2)."""
    H, kp1, kp2, matches = get_homography(img1, img2)
    if H is None: return img2
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, H)
    
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    output_img = cv2.warpPerspective(img1, translation @ H, (x_max - x_min, y_max - y_min))
    output_img[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2
    
    return output_img

img1 = cv2.imread('./images/gambar1.jpg') 
img2 = cv2.imread('./images/gambar2.jpg')
img3 = cv2.imread('./images/gambar3.jpg') 
step1_res = stitch_images(img1, img2)
final_panorama = stitch_images(step1_res, img3)


plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); plt.title("Img 1 (Left)"); plt.axis('off')
plt.subplot(3, 3, 2); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); plt.title("Img 2 (Center)"); plt.axis('off')
plt.subplot(3, 3, 3); plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)); plt.title("Img 3 (Right)"); plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(cv2.cvtColor(step1_res, cv2.COLOR_BGR2RGB))
plt.title("Step 1: Merging Image 1 & 2")
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
plt.title("Final Step: Full 3-Image Panorama")
plt.axis('off')

plt.tight_layout()
plt.savefig('06_multi_panorama_steps.png')
plt.show()