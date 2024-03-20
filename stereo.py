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
                
            mean = np.mean(copy_patch, axis=2, keepdims=True)
            # print(mean.shape)
            copy_patch -= mean # cp- H x W x 3 (r,g,b)
            flat_array = copy_patch.reshape(-1)
            normalize = flat_array/np.linalg.norm(flat_array)
            normalized_image[y,x] = normalize
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
    right_descriptors = get_ncc_descriptors(img_right, patchsize)
    left_descriptors = get_ncc_descriptors(img_left, patchsize)
    print("TRight Shape:" + str(right_descriptors.shape))
    print("TLeft Shape:" + str(left_descriptors.shape))
    

    
    # normalized images  [:,d:,:]
   
    for d in range(dmax): # should include dmax as well

        right_sliced = right_descriptors[:,:r_width-d,:]
        left_sliced = left_descriptors[:,d:,:]
        # print("TRightSLiced Shape:" + str(right_sliced.shape))
        # print("TLeftSLiced Shape:" + str(left_sliced.shape))
        ncc_vol[d,:,:r_width-d]= np.sum(right_sliced*left_sliced, axis=2)
        # #shift left descriptor 
        # for y in range(r_height):
        #     for x in range(r_width): # the left image is offset by d: (x+d, y)
        #         #ensure index is in bounds
        #         dot_prod = right_sliced[y, x] * left_sliced[y, x]
        #         ncc_vol[d, y, x] = dot_prod  # Use 
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





    
