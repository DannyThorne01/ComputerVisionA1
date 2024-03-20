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

    height, width, channels = img.shape
    descriptors = np.zeros((height, width, channels * patchsize**2))
    for y in range(height):
        for x in range(width):
            start_patch_y, end_patch_y= max(0, y - patchsize // 2), min(height, y + patchsize // 2 + 1)
            start_patch_x,end_patch_x= max(0, x - patchsize // 2), min(width, x + patchsize // 2 + 1)

            # Check if the patch extends beyond the image boundaries
            if start_patch_y == 0 or end_patch_y == height or start_patch_x == 0 or end_patch_x == width:
                # If so, keep the descriptor as zeros.
                continue

            patch = img[start_patch_y :end_patch_y , start_patch_x: end_patch_x, :]

            patch_mean = np.mean(patch, axis=(0, 1))
            patch -= patch_mean
            flat_patch = patch.reshape(-1)
            norm = np.linalg.norm(flat_patch)
            descriptors[y, x, :] = flat_patch / norm

    return descriptors

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
   
    for d in range(dmax): 

        right_sliced = right_descriptors[:,:r_width-d,:]
        left_sliced = left_descriptors[:,d:,:]
       
        ncc_vol[d,:,:r_width-d]= np.sum(right_sliced*left_sliced, axis=2)
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






    
