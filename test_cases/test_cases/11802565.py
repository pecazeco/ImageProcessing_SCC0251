###### Assignment: enhancement and superresolution #####
# SCC0251 - Image Processing and Analysis - 2024/1
# Pedro Azevedo Coelho Carriello Correa - No USP: 11802565


#### Importing libraries ####

import numpy as np
import imageio.v3 as img


#### Enhancement Functions' Definitions ####
 
def jointCumulativeHistogram(low_img):
    hist_c_joint = hc(low_imgs_arr) # cumulative histogram of all the 4 low resolution images
    L = int(low_imgs_arr.max()) + 1 # the number of intensities
    M,N = low_img.shape # image resolution
    pixels_num = 4*M*N # number of pixels considering all 4 images
    
    img_enhanced = np.copy(low_img) # inicializing the enhanced version of the low resolution image as a copy of the original one
    for i in range(len(img_enhanced)):
        for j in range(len(img_enhanced[i])):
            pixel = int(img_enhanced[i,j])
            img_enhanced[i,j] = (L-1)/pixels_num * hist_c_joint[pixel] # applying histogram equalisation transformation 
    
    return img_enhanced
    
def singleImageCumulativeHistogram(low_img):
    hist_c = hc(low_img) # cumulative histogram of the single image
    L = low_img.max() + 1 # number of intensities
    M,N = low_img.shape # image resolution
    
    img_enhanced = np.copy(low_img) # inicializing the enhanced version of the low resolution image as a copy of the original one
    for i in range(len(img_enhanced)):
        for j in range(len(img_enhanced[i])):
            pixel = int(img_enhanced[i,j])
            img_enhanced[i,j] = (L-1)/(M*N) * hist_c[pixel] # applying transformation 
    
    return img_enhanced

def gammaCorrection(low_img): 
    img_enhanced = np.floor( 255 * np.power(low_img/255, 1/gamma) ) # applying gamma correction method for the low resolution image
    return img_enhanced
    
    
#### Auxiliary Functions ####

def histogram(img):
    num_levels = int(img.max()) + 1 # number of intensities levels
    
    hist = np.zeros(num_levels, dtype=int) 
    for i in range(num_levels):
        hist[i] = np.sum(img == i) # calculating histogram
    
    return hist

def hc(img): # function for compute cumulative histogram
    num_levels = int(img.max()) + 1 # number of intensities levels
    hist = histogram(img) # calculating standard histogram
    
    hist_c = np.zeros(num_levels, dtype=int) # inicializing cumulative histogram
    hist_c[0] = hist[0]
    for i in range(1,num_levels):
        hist_c[i] = hist_c[i-1] + hist[i] # cumulative histogram
    
    return hist_c
    
def compose_superresolution(L): # function for compose the superresolution image from the 4 low resolution original images
    l1,l2,l3,l4 = L # from a array of the 4 low resolution images, create a variable for each of the images
    M,N = l1.shape # resolution of the original images
    
    H = np.zeros((2*M, 2*N)) # inicializing the superresolution version 
    for i_M in range(M): # composes the new version from the original ones
        for j_N in range(N):
            H[2*i_M, 2*j_N] = l1[i_M,j_N] 
            H[2*i_M+1, 2*j_N] = l2[i_M,j_N]
            H[2*i_M, 2*j_N+1] = l3[i_M,j_N]
            H[2*i_M+1, 2*j_N+1] = l4[i_M,j_N]
    
    return H
    
def RMSE(H, H_hat): # funcion for calculate the root mean square error
    N,M = H.shape # resolution of the images
    rmse = np.sqrt( np.sum( np.power( H-H_hat, 2) ) / (N*M) ) # 
    return rmse


#### Main Code ####

## Inputs of the images and required parameters
imglow = input().rstrip()
size_low = img.imread(imglow + '0.png').shape # shape of the low resolution images to use as a parameter for inicializing the array of these images
low_imgs_arr = np.zeros(np.append(4, size_low)) # inicialing the array that containes the low resolution images
for i in range(4):
     low_imgs_arr[i] = img.imread(imglow + str(i) + '.png') # filling in the array with the input images

imghigh = input().rstrip()
imghigh = img.imread(imghigh) # the high resolution input image

F = input().rstrip() # the parameter that chooses the method of enhancement
gamma = int(float(input().rstrip())) 


low_imgs_arr_enhanced = np.copy(low_imgs_arr) # inicializing the array of the enhanced versions 

## the switch cases for each option F
if F=='0':
    pass # that option doesnt do anything 
elif F=='1':
    for i in range(4):
        low_imgs_arr_enhanced[i] = singleImageCumulativeHistogram(low_imgs_arr[i]) 
elif F=='2':
    for i in range(4):
        low_imgs_arr_enhanced[i] = jointCumulativeHistogram(low_imgs_arr[i])    
elif F=='3':
    for i in range(4):
        low_imgs_arr_enhanced[i] = gammaCorrection(low_imgs_arr[i])        
else:
    raise ValueError('choose a numerical value for F among 0,1,2,3')

img_enhanced_superresolution = compose_superresolution(low_imgs_arr_enhanced) # this is the superresolution version of the enhanced versions of the original low resolution images


error = RMSE(imghigh, img_enhanced_superresolution) # calculates the error from the original high quality image and the superresolution enhanced version
print(error)