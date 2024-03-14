import numpy as np
#==============No additional imports allowed ================================#

def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32 (50,50,3)
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor
    
    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    
    height,width,channels = img.shape #find the specific order

    # Stores a copy of the original image
    normalized_image = np.zeros([height, width,(channels * patchsize**2)])  
    for y in range(height):
        for x in range(width):
            # Get start and end values of the patch
            # print(patchsize) 
            # StartX : -3.0
            # EndX: 3.0
            # StartY : -3.0
            # EndY: 3.0
            start_x = x - (patchsize-1)//2
            end_x   = x+ (patchsize-1)//2
            # print("StartX : " + str(start_x))
            # print("EndX: " + str(end_x))
            start_y = y - (patchsize-1)//2
            end_y   = y + (patchsize-1)//2
            # print("StartY : " + str(start_y))
            # print("EndY: " + str(end_y))
            r_sum=0
            g_sum=0
            b_sum=0

            copy_patch = np.zeros([patchsize,patchsize,3])
            #make a copy patch of fixed size (7,7,3) no matter if part of original patch OOB
            for inner_y in range(start_y,end_y):
                for inner_x in range(start_x, end_x):
                    offset = (patchsize-1)//2
                 
                    # if patch goes out of bounds zero the value
                    if (inner_x<0 or inner_x >= width) or (inner_y<0 or inner_y>=height):

                        copy_patch[(inner_y+offset)%patchsize][(inner_x+offset)%patchsize] = 0
                    # else "copy" pixel value from original image into patch
                    else: 
                        copy_patch[(inner_y+offset)%patchsize][(inner_x+offset)%patchsize]= img[inner_y][inner_x]
                    
            # Check if there exists some easier way to do this in numpy
            # Sums up the channels for each pixel
            #TypeError: 'float' object cannot be interpreted as an integer
            for patch_y in range(0,patchsize):
                for patch_x in range(0,patchsize):
                    r_sum += copy_patch[patch_y][patch_x][0] # gives R for this pixel
                    g_sum += copy_patch[patch_y][patch_x][1] # gives G for this pixel
                    b_sum += copy_patch[patch_y][patch_x][2] # gives B for this pixel

            # Compute the mean for every channel
            patch_num_pixels = (patchsize)**2   
            r_avg = r_sum / patch_num_pixels
            g_avg = g_sum / patch_num_pixels
            b_avg = b_sum / patch_num_pixels

            # NOTE: subtract the mean of each channel from every pixel in the patch - what if mean is greater than actual value ?
            for patch_y in range(0,patchsize):
                for patch_x in range(0,patchsize): # SWAP THE OTHER WAY
                    copy_patch[patch_y][patch_x][0] -= r_avg
                    copy_patch[patch_y][patch_x][1] -= g_avg
                    copy_patch[patch_y][patch_x][2] -= b_avg
                
            # if start_x < 0: 
            #     start_x = 0
            # if end_x >= width: 
            #     end_x = width - 1
            # if start_y < 0: 
            #     start_y = 0
            # if end_y >= height: 
            #     end_y = height - 1

            # print("New StartX : " + str(start_x))
            # print("New EndX: " + str(end_x))
            # print("New StartY : " + str(start_y))
            # print("New EndY: " + str(end_y))
            
            #so we flatten from img[start_x:end_x][start_y:end_y] into one dimension array
            # patch_list = []
            # for inner_y in range(start_y,end_y+1):
            #     patch_list.append(img[inner_y][start_x:end_x+1])
            # print(patch_list[0])
            # calc for first 
            # temp = img[start_y:end_y+1,start_x:end_x+1] --> (4,4,3)->(27,) but we want (7,7,3)->(147,) OOB zeroed
            # print(temp)
            # temp = copy_patch.flatten()
            # print(temp.shape)
            flat_array = copy_patch.flatten()
            # print("Flat Array : ", str(flat_array))
            # print(flat_array.shape)

            normalize = flat_array/np.linalg.norm(flat_array)
            # print(normalize)
            # print(normalize.shape)
            normalized_image[y][x] = normalize
    return normalized_image
            

def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
    r_height,r_width,r_channel = img_right.shape
    l_height,l_width,l_channel = img_left.shape
    
    ncc_vol = np.zeros([dmax, r_height, r_width])
    
    # normalized images
    normalized_img_right = get_ncc_descriptors(img_right, patchsize)
    normalized_img_left = get_ncc_descriptors(img_left, patchsize)

    for y in range(r_height):
        for x in range(r_width): # the left image is offset by d: (x+d, y)
            # Get start and end values of the patch
            start_x = x - (patchsize-1)//2
            end_x   = x+ (patchsize-1)//2

            start_y = y - (patchsize-1)//2
            end_y   = y + (patchsize-1)//2
            
            normalized_patch_img_right = normalized_img_right[start_y:end_y + 1][start_x:end_x+1]
            # might have to bound start_x+dmax:end_x+1+dmax in case they go too far out
            normalized_patch_img_left = normalized_img_left[start_y:end_y + 1][start_x+dmax:end_x+1+dmax]
            print("TRight Shape:" + str(normalized_patch_img_right.shape))
            print("TLeft Shape:" + str(normalized_patch_img_left.shape))
            # dot product between patches
            dot_product = np.dot(normalized_patch_img_right,normalized_patch_img_left.T)
            ncc_vol[dmax][y][x] = dot_product
    return ncc_vol

            
        
def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax x H x W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    print(ncc_vol[0])





    
