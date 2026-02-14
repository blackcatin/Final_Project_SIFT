import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/gambar1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

k = max(kp, key=lambda x: x.size)
x, y, s = int(k.pt[0]), int(k.pt[1]), int(k.size)

patch = gray[y-8:y+8, x-8:x+8]
if patch.shape == (16, 16):
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    plt.figure(figsize=(8, 8))
    X, Y = np.meshgrid(np.arange(16), np.arange(16))
    plt.quiver(X, Y, gx, -gy, color='red') 
    plt.title("Local Gradient Vector Field (Descriptor Basis)")
    plt.gca().invert_yaxis()
    plt.savefig('04_descriptor_vector.png')
    plt.show()