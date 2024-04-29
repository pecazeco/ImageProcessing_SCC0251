#### Importing libraries ####

import numpy as np
import imageio.v3 as im
import matplotlib.pyplot as plt


#### Functions ####

def grayscale(img):
    gray_transform = np.array([ .2989, .5870, .1140])
    return np.dot(img, gray_transform).astype(np.int64)
    return img[:,:,0]

def thresholding(f, L):
    f_tr = np.ones(f.shape).astype(np.uint8)
    f_tr[ np.where( f<L ) ] = 0
    return f_tr

def otsu(img, max_L):
    M = np.prod(img.shape)
    min_var = []
    hist_t,_ = np.histogram(img, bins=256, range=(0,256))
    img_t = thresholding(img, 0)
    for L in np.arange(1, max_L):
        img_ti = thresholding(img, L)
        #computing weights
        w_a = np.sum(hist_t[:L])/float(M)
        w_b = np.sum(hist_t[L:])/float(M)
        #computing variances
        if np.where(img_ti == 0)[0].size == 0 or np.where(img_ti == 1)[0].size == 0:
            min_var = min_var + [np.inf]
            continue
        sig_a = np.var(img[np.where(img_ti == 0)])
        sig_b = np.var(img[np.where(img_ti == 1)])
        
        min_var = min_var + [w_a*sig_a + w_b*sig_b]
        
    img_t = thresholding(img, np.argmin(min_var))

    return img_t, np.argmin(min_var)

def dilation(img):
    x_indx, y_indx = np.where( img==1 )
    mask = np.copy(img)
    for x,y in zip(x_indx, y_indx):
        if (x==0 or x==img.shape[0]-1) or (y==0 or y==img.shape[1]-1):
            continue
        else:
            for i in (x-1,x,x+1):
                for j in (y-1,y,y+1):
                    mask[i,j] = 1
    
    return mask
    
def erosion(img):
    x_indx, y_indx = np.where( img==1 )
    mask = np.copy(img)
    for x,y in zip(x_indx, y_indx):
        if (x==0 or x==img.shape[0]-1) or (y==0 or y==img.shape[1]-1):
            continue
        else:
            key = 1
            for i in (x-1,x,x+1):
                for j in (y-1,y,y+1):
                    if img[i,j]==0:
                        key = 0
                        break
                if key==0:
                    break
            if key == 0:
                mask[x,y] = 0
    
    return mask

def filter_gaussian(P, Q):
    s1 = P
    s2 = Q

    D = np.zeros([P, Q])  # Compute Distances
    for u in range(P):
        for v in range(Q):
            x = (u-(P/2))**2/(2*s1**2) + (v-(Q/2))**2/(2*s2**2)
            D[u, v] = np.exp(-x)
    
    return D

def map_value_to_color(value, min_val, max_val, colormap):
    # Scale the value to the range [0, len(colormap) - 1]
    scaled_value = (value - min_val) / (max_val - min_val) * (len(colormap) - 1)
    # Determine the two closest colors in the colormap
    idx1 = int(scaled_value)
    idx2 = min(idx1 + 1, len(colormap) - 1)
    # Interpolate between the two colors based on the fractional part
    frac = scaled_value - idx1
    color = [
        (1 - frac) * colormap[idx1][0] + frac * colormap[idx2][0],
        (1 - frac) * colormap[idx1][1] + frac * colormap[idx2][1],
        (1 - frac) * colormap[idx1][2] + frac * colormap[idx2][2]
    ]
    
    return color

def RMSE(img, img_hat): # Root mean squared error
    m,n = img.shape
    error = np.sqrt( 1/(m*n) * np.sum( np.power(img - img_hat,2) ) )
    
    return error


#### Main code ####

## Set images and indexes ##

filename_img_input = 'test_cases_data/' + input().rstrip()
img_input = im.imread(filename_img_input)

filename_img_reference = 'test_cases_references/' + input().rstrip()
img_reference = im.imread(filename_img_reference)

indexes_technique = input().split()

img_input_gray = grayscale(img_input) if len(img_input.shape)==3 else np.copy(img_input) # transform to gray scale if the image is not already gray scaled

img_input_binarized, L = otsu(img_input_gray, 200)

img_input_mask = np.copy(img_input_binarized) # initializing the mask
for index in indexes_technique: 
    
    if index == '1': # Chooses Erosion
        img_input_mask = erosion(img_input_mask)
        
    elif index == '2': # Chooses Dilation
        img_input_mask = dilation(img_input_mask)
        
    else: # Wrong index for technique 
        raise ValueError('Choose the indexes among 1 and 2 for the techniques')
    
M,N = img_input.shape[0:2]
filter = filter_gaussian(M,N)

min_val = np.min(np.array(filter))
max_val = np.max(np.array(filter))

#Espectro Visível
heatmap_colors = [
    [1, 0, 1],   # Pink
    [0, 0, 1],   # Blue
    [0, 1, 0],   # Green
    [1, 1, 0],   # Yellow
    [1, 0, 0]    # Red
]

filter_heatmap = np.zeros([M, N, 3]) #Imagem RGB vazia
for i in range(M):
    for j in range(N):
        filter_heatmap[i, j] = map_value_to_color(filter[i, j], min_val, max_val, heatmap_colors)

img_input_mask_color = np.ones([M, N, 3]) #Imagem RGB vazia
indexes = np.where(img_input_mask==0)
img_input_mask_color[indexes] = filter_heatmap[indexes]

img_input_gray_normalized = img_input_gray / np.max(img_input_gray)

alpha = 0.3
img_output_mixed = np.zeros_like(img_input_mask_color)
for i in range(3):
    img_output_mixed[:,:,i] = (1-alpha) * img_input_gray_normalized + alpha * img_input_mask_color[:,:,i]


#### Calculate error ####

## Normalizing the restored image to the range 0 to 255
max_pixel = np.max(img_output_mixed)
min_pixel = np.min(img_output_mixed)
img_output_normalized = np.zeros_like(img_output_mixed, dtype=np.float64)
M,N = img_output_mixed.shape[0:2]
for i in range(M):
    for j in range(N):
        for k in range(3):
            pixel = img_output_mixed[i,j,k]
            img_output_normalized[i,j,k] = 255 * (pixel - min_pixel) / (max_pixel - min_pixel) # normalizing each pixel

## Calculating error for each channel 
error_R = RMSE(img_output_normalized[:,:,0], img_reference[:,:,0])
error_G = RMSE(img_output_normalized[:,:,1], img_reference[:,:,1])
error_B = RMSE(img_output_normalized[:,:,2], img_reference[:,:,2])

## Resulting error
error = (error_R + error_G + error_B)/3
print(f"Error: {error:.4f}")


#### Plotting the images ####

fig, axarr = plt.subplots(4,2)

axarr[0,0].imshow(img_input)
axarr[0,0].set_title('original input image')

axarr[0,1].imshow(img_input_gray, cmap='gray')
axarr[0,1].set_title('gray scaled input image')

axarr[1,0].imshow(img_input_binarized, cmap='gray')
axarr[1,0].set_title('binarized input image')

axarr[1,1].imshow(img_input_mask, cmap='gray')
axarr[1,1].set_title('binary mask')

axarr[2,0].imshow(filter, cmap='gray')
axarr[2,0].set_title('gaussian filter')

axarr[2,1].imshow(filter_heatmap)
axarr[2,1].set_title('gaussian heatmap')

axarr[3,0].imshow(img_input_mask_color)
axarr[3,0].set_title('colored mask')

axarr[3,1].imshow(img_output_mixed)
axarr[3,1].set_title('output image')

fig.set_figwidth(7)
fig.set_figheight(14)
fig.suptitle("Processing's step-by-step", fontsize=25)

plt.savefig('transformation_plots/' + filename_img_input[16:])