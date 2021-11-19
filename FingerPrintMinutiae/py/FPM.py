import numpy as np
import cv2
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
import Minutiae_Cikarici
from Minutiae_Cikarici import extract_minutiae_features
import math

#Burada 
img = cv2.imread("Image/AFP.bmp",0)


def Mythresold(image):
    thresh = threshold_otsu(image)
    binary = image > thresh

    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')

    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')

    plt.show()


thresh = 70

ret,th = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((3,3),np.uint8)

erosion = cv2.erode(th,kernel,iterations=1)
medyan = cv2.medianBlur(erosion,3)
dilation = cv2.dilate(medyan,kernel,iterations=1)
dilation = cv2.medianBlur(dilation,3)
dilation = dilation/255
skeleton = skeletonize(dilation)
skeleton = invert(skeleton)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()

ridge_terminations_kernel = [
    np.array([[-1, 1,-1],
              [-1, 1,-1],
              [-1,-1,-1]], dtype=np.int),

    np.array([[-1,-1,-1],
              [-1, 1, 1],
              [-1,-1,-1]], dtype=np.int),

    np.array([[-1,-1,-1],
              [-1, 1,-1],
              [-1, 1,-1]], dtype=np.int),
    
    np.array([[-1,-1,-1],
              [ 1, 1,-1],
              [-1,-1,-1]], dtype=np.int),
    
    np.array([[ 1,-1,-1],
              [-1, 1,-1],
              [-1,-1,-1]], dtype=np.int),
        
    np.array([[-1,-1, 1],
              [-1, 1,-1],
              [-1,-1,-1]], dtype=np.int),
    
    np.array([[-1,-1,-1],
              [-1, 1,-1],
              [-1,-1, 1]], dtype=np.int),
    
    np.array([[-1,-1,-1],
              [-1, 1,-1],
              [ 1,-1,-1]], dtype=np.int)]

ridge_bifurcations_kernel = [
    np.array([[-1,-1, 1],
              [ 1, 1,-1],
              [-1,-1, 1]], dtype=np.int),

    np.array([[-1, 1,-1],
              [-1, 1,-1],
              [ 1,-1, 1]], dtype=np.int),

    np.array([[ 1,-1,-1],
              [-1, 1, 1],
              [ 1,-1,-1]], dtype=np.int),
    
    np.array([[ 1,-1, 1],
              [-1, 1,-1],
              [-1, 1,-1]], dtype=np.int),
    
    np.array([[-1, 1,-1],
              [-1, 1, 1],
              [ 1,-1,-1]], dtype=np.int),
        
    np.array([[ 1,-1,-1],
              [-1, 1, 1],
              [-1, 1,-1]], dtype=np.int),
    
    np.array([[-1,-1, 1],
              [ 1, 1,-1],
              [-1, 1,-1]], dtype=np.int),
    
    np.array([[-1, 1,-1],
              [ 1, 1,-1],
              [-1,-1, 1]], dtype=np.int)]

def extract_ridge(input_img, kernel):
    img = input_img.copy()
    result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(kernel)):
        result = cv2.bitwise_or(result, cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel[i]))
    return result  

def manhattan_distance(x, y):
    (x1,x2)=x
    (y1,y2)=y
    return abs(x1 - y1) + abs(x2 - y2)

def euclidean_distance(x, y):
    (x1,x2)=x
    (y1,y2)=y
    return math.sqrt((x1 - y1)**2 + (x2 - y2)** 2)

def get_minutiae_list(minutiae_img):
    minutiae = []
    (iMax, jMax) = minutiae_img.shape
    for i in range(iMax):
        for j in range(jMax):
            if minutiae_img[i][j] != 0:
                minutiae.append((i,j))
    return minutiae

def postprocessing_minutiae_list_same_type(minutiae_list, tresh = 8, dist = manhattan_distance):
    res = minutiae_list.copy()
    for m1 in minutiae_list:
        for m2 in minutiae_list:
            if m1 != m2 and dist(m1, m2) < tresh:
                if m1 in res:
                    res.remove(m1)
                if m2 in res:
                    res.remove(m2)
    return res

def postprocessing_minutiae_list_diff_type(minutiaeList1, minutiaeList2, tresh = 8, dist = manhattan_distance):
    res = minutiaeList1.copy()
    res = res + minutiaeList2
    for m1 in minutiaeList1:
        for m2 in minutiaeList2:
            if dist(m1, m2) < tresh:
                if m1 in res:
                    res.remove(m1)
                if m2 in res:
                    res.remove(m2)
    return res

def postprocessing(terminations_img, bifurcations_img, tresh1 = 4, tresh2 = 4, tresh3 = 4, dist = manhattan_distance):
    terminations_list = get_minutiae_list(terminations_img)
    bifurcations_list = get_minutiae_list(bifurcations_img)
    terminations_list = postprocessing_minutiae_list_same_type(terminations_list, tresh1, dist)
    bifurcations_list = postprocessing_minutiae_list_same_type(bifurcations_list, tresh2, dist)
    result_list = postprocessing_minutiae_list_diff_type(terminations_list, bifurcations_list, tresh3, dist)
    iMax, jMax = terminations_img.shape
    result_img = np.zeros((iMax, jMax), dtype=np.uint8)
    for (i,j) in result_list:
        result_img[i][j] = 1
    return result_img, result_list

# if __name__ == "__main__":
#     img = cv2.imread('Skeleton.png', cv2.IMREAD_GRAYSCALE)
#     terminations = extract_ridge(img, ridge_terminations_kernel)
#     bifurcations = extract_ridge(img, ridge_bifurcations_kernel)
#     fin_img, fin_list = postprocessing(terminations, bifurcations)
#     show_minutiae = img.copy()
#     show_minutiae = cv2.cvtColor(show_minutiae, cv2.COLOR_GRAY2BGR)
#     for (i,j) in fin_list:
#         cv2.circle(show_minutiae,(j,i), 4, (255,0,0), cv2.LINE_4)
    

if __name__ == '__main__':
    img = cv2.imread('SSkeleton.png',0)
    FeaturesTerminations, FeaturesBifurcations = Minutiae_Cikarici.extract_minutiae_features(img, showResult=True)
    

cv2.waitKey(0)
cv2.destroyAllWindows()
