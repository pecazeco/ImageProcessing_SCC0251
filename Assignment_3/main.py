#### Importing libraries ####

import numpy as np
import imageio.v3 as im
import matplotlib.pyplot as plt


#### Functions ####

def grayscale(img):
    return np.dot(img, [ .2989, .5870, .1140]).astype(np.int64)


#### Main code ####

## Set images and indexes ##

filename_img_input = 'test_cases_data/' + input().rstrip()
img_input = im.imread(filename_img_input)

filename_img_reference = 'test_cases_references/' + input().rstrip()
img_reference = im.imread(filename_img_reference)

indexes_technique = input().rstrip()
indexes_technique = indexes_technique.split()

img_input_gray = grayscale(img_input) # transform to gray scale


#### Plotting the images ####

fig, axarr = plt.subplots(1,2)
axarr[0].imshow(img_input)
axarr[1].imshow(img_input_gray, cmap='gray')
# axarr[2].imshow(filter, cmap='gray')
# axarr[3].imshow(restored_img, cmap='gray')

plt.savefig('output_' + filename_img_input)

