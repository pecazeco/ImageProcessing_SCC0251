'''
SCC0251 - Image Processing and Analysis - 2024/1
Assignment: Fourier Transform & Filtering in Frequency Domain
'''


#### Importing libraries ####

import numpy as np
import imageio.v3 as im
import matplotlib.pyplot as plt


#### Functions ####

## Functions for make the filters ## 

def highpass(img, r): # Ideal High-pass Filter
    H = np.zeros(img.shape) 
    P,Q = img.shape
    
    for u in range(P):
        for v in range(Q):
            D = np.sqrt( np.power(u-P/2 ,2) + np.power(v-Q/2 ,2) ) 
            H[u,v] = 0 if D <= r else 1  # If we are outide a given circle, the frequency pass
    
    return H
    
def lowpass(img, r): # Ideal Low-pass Filter
    H = np.zeros(img.shape)
    P,Q = img.shape
    
    for u in range(P):
        for v in range(Q):
            D = np.sqrt( np.power(u-P/2 ,2) + np.power(v-Q/2 ,2) )
            H[u,v] = 1 if D <= r else 0  # If we are inside a given circle, the frequency pass
    
    return H

def bandstop(img, r0, r1): # Ideal Band-stop filter
    H = np.zeros(img.shape)
    P,Q = img.shape
    
    for u in range(P):
        for v in range(Q):
            D = np.sqrt( np.power(u-P/2 ,2) + np.power(v-Q/2 ,2) )
            H[u,v] = 0 if ( D>=r1 and D<=r0 ) else 1  # the frequency just pass if its not on the area of a given ring
    
    return H

def laplacian_highpass(img): # Laplacian High-pass filter
    H = np.zeros(img.shape)
    P,Q = img.shape
    
    for u in range(P):
        for v in range(Q):
            H[u,v] = -4*np.power(np.pi,2) * ( np.power( u-P/2, 2 ) + np.power( v-Q/2, 2 ) ) # equation of the frequency response of the laplacian low-pass filter
    
    return H

def gaussian_lowpass(img, lambda1, lambda2): # Gaussian low-pass filter 
    H = np.zeros(img.shape)
    P,Q = img.shape
    
    for u in range(P):
        for v in range(Q):
            x = (u-P/2)**2/(2*lambda1**2) + (v-Q/2)**2/(2*lambda2**2) # equation of the frequency response of the gaussian filter considering the given standard deviations
            H[u,v] = np.exp(-x)
    
    return H

## Auxiliary Functions ##

def RMSE(img, img_hat): # Root mean squared error
    m,n = img.shape
    error = np.sqrt( 1/(m*n) * np.sum( np.power(img - img_hat,2) ) )
    
    return error


#### Main Code ####

## Inputs the filename of the input image and the reference
input_img = im.imread('Files_provided_by_the_teacher/test_cases_data/' + input().rstrip())
reference_img_filename = input().rstrip()
reference_img = im.imread('Files_provided_by_the_teacher/test_cases_reference/' + reference_img_filename)

## Inputs the index of which filter is gonna be used
filter_index = input().rstrip() 

## Fourier transformation of the input image 
input_img_fourier = np.fft.fft2(input_img) # fourier transform
input_img_fourier = np.fft.fftshift(input_img_fourier) # shift version of the transform

## Switch cases for the creation of filter and its respective parameters inputs ##

if filter_index == '0':
    r = int(input().rstrip())
    filter = lowpass(input_img_fourier, r)
    
elif filter_index == '1':
    r = int(input().rstrip())
    filter = highpass(input_img_fourier, r)
    
elif filter_index == '2':
    r0 = int(input().rstrip())
    r1 = int(input().rstrip())
    filter = bandstop(input_img_fourier, r0, r1)
    
elif filter_index == '3':
    filter = laplacian_highpass(input_img_fourier)
    
elif filter_index == '4':
    lambda1 = int(input().rstrip())
    lambda2 = int(input().rstrip())
    filter = gaussian_lowpass(input_img_fourier, lambda1, lambda2)

else:
    raise ValueError('choose a numerical value for the filter index between 0 and 4')

## Creating the restored image
restored_img = input_img_fourier * filter 
restored_img = np.fft.ifftshift(restored_img)  # inverse shift 
restored_img = np.fft.ifft2(restored_img)  # inverse Fourier Transform
restored_img = np.real(restored_img)  # pick the real part as the restored image

## Normalizing the restored image to the range 0 to 255
max_pixel = np.max(restored_img)
min_pixel = np.min(restored_img)
normalized_restored_img = np.zeros_like(restored_img, dtype=np.float64)
M,N = restored_img.shape
for i in range(M):
    for j in range(N):
        pixel = restored_img[i,j]
        normalized_restored_img[i,j] = 255 * (pixel - min_pixel) / (max_pixel - min_pixel) # normalizing each pixel

## Calculating error by RMSE
error = RMSE(normalized_restored_img, reference_img) 
print(error)


#### Save plots as figures ####

fig, axarr = plt.subplots(1,4)
axarr[0].imshow(input_img, cmap='gray')
axarr[1].imshow(np.log(1 + np.abs(input_img_fourier)), cmap='gray')
axarr[2].imshow(filter, cmap='gray')
axarr[3].imshow(restored_img, cmap='gray')

plt.savefig('output_plots/output_case' + reference_img_filename.split('case')[-1])