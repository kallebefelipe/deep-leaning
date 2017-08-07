import matplotlib.pyplot as plt
import cv2
img = cv2.imread('1000042_cut_797575.png', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
cl1 = clahe.apply(img)

cv2.imwrite('clahe_2.jpg', cl1)
plt.imshow(cl1)
