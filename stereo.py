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
            start_x = x - (patchsize-1)//2
            end_x   = x+ (patchsize-1)//2
            start_y = y - (patchsize-1)//2
            end_y   = y + (patchsize-1)//2
            r_sum=0
            g_sum=0
            b_sum=0

            copy_patch = np.zeros([patchsize,patchsize,3])
            #make a copy patch of fixed size (7,7,3) no matter if part of original patch OOB
            # -3, -2, -1, 0, 1, 2, 3 -> 0, 1, 2, 3, 4, 5, 6 
            for inner_y in range(start_y,end_y+1):
                for inner_x in range(start_x, end_x+1):
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
            # for patch_y in range(0,patchsize):
            #     for patch_x in range(0,patchsize):
            #         r_sum += copy_patch[patch_y][patch_x][0] # gives R for this pixel
            #         g_sum += copy_patch[patch_y][patch_x][1] # gives G for this pixel
            #         b_sum += copy_patch[patch_y][patch_x][2] # gives B for this pixel

            # # Compute the mean for every channel
            # patch_num_pixels = (patchsize)**2   
            # r_avg = r_sum / patch_num_pixels
            # g_avg = g_sum / patch_num_pixels
            # b_avg = b_sum / patch_num_pixels
             # Subtract mean
                        
            #  # NOTE: subtract the mean of each channel from every pixel in the patch - what if mean is greater than actual value ?
            # for patch_y in range(0,patchsize):
            #     for patch_x in range(0,patchsize): # SWAP THE OTHER WAY
            #         copy_patch[patch_y][patch_x][0] -= r_avg
            #         copy_patch[patch_y][patch_x][1] -= g_avg
            #         copy_patch[patch_y][patch_x][2] -= b_avg
            # flat_array = copy_patch.flatten()
            # normalize = flat_array/np.linalg.norm(flat_array)
            # normalized_image[y][x] = normalize
                        
            mean = np.mean(copy_patch, axis=2, keepdims=True)
            print(mean.shape)
            copy_patch -= mean # cp- H x W x 3 (r,g,b)
            flat_array = copy_patch.flatten()
            normalize = flat_array/np.linalg.norm(flat_array)
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
    r_height,r_width,_ = img_right.shape
    
    ncc_vol = np.zeros([dmax, r_height, r_width])
    
    # normalized images
    right_decriptors = get_ncc_descriptors(img_right, patchsize)
    left_decriptors = get_ncc_descriptors(img_left, patchsize)
    print("TRight Shape:" + str(right_decriptors.shape))
    print("TLeft Shape:" + str(left_decriptors.shape))
    # result = np.dot(right_decriptors[1][1],left_decriptors[1][1+1])
    # result1 = np.dot(right_decriptors[1,1], left_decriptors[1,1])
    # print(result1.shape)
    # print(result.shape)
    # print(result)
    # print(result1)
    # print(dmax)
    # print(ncc_vol[1][2][3].shape)
    # ncc_vol[1][2][3]= result 
    ncc_vol = np.zeros([dmax, r_height, r_width])
    
    # normalized images
    right_descriptors = get_ncc_descriptors(img_right, patchsize)
    left_descriptors = get_ncc_descriptors(img_left, patchsize)
    for d in range(0,dmax): # should include dmax as well
        for y in range(r_height):
            for x in range(r_width): # the left image is offset by d: (x+d, y)
                #ensure index is in bounds
                if(x+d <r_width):
                    dot_prod = np.dot(right_descriptors[y, x], left_descriptors[y, x+d])
                    ncc_vol[d, y, x] = dot_prod  # Use 
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
    max_indx = np.argmax(ncc_vol,axis=0)
    return max_indx
    # return np.max(ncc_vol, axis=0)





    
