import numpy as np
##======================== No additional imports allowed ====================================##






def photometric_stereo_singlechannel(I, L):

    #L is 3 x k : Direction to the light source
    #I is k x n : Intrinsic Image decomposition
    # G is  3 x n : 
   
    G = np.linalg.inv(L @ L.T) @ L @ I 
   
    # Albedo (p) = norm of G -> ||G||
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals


def photometric_stereo(images, lights):
    '''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N  images, each a numpy float array of size H x W x 3- NxHxWx3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''
    # just need to compute I and L and then we pass those through to helper 
    # we get I from G and G is just all the light sources I =G.transpose * L
    height,width,_ = images[0].shape
    N = len(images)
    red = np.zeros((N,(height*width)))
    green = np.zeros((N,(height*width)))
    blue = np.zeros((N, (height*width)))

    for i, image  in enumerate(images):
        red[i] = image[:, :, 0].flatten()
        green[i] = image[:, :, 1].flatten()
        blue[i] = image[:, :, 2].flatten()

    # Process each color channel through photometric_stereo_singlechannel
    albedos = []
    normals_list = []
    for channel_data in (red, green, blue):
        albedo_channel, normals_channel = photometric_stereo_singlechannel(channel_data, lights)
        albedos.append(albedo_channel)
        normals_list.append(normals_channel)

    avg_normals = np.mean(normals_list,axis = 0)
    norms = np.linalg.norm(avg_normals, axis=0, keepdims=True)
    unit_normals = avg_normals / norms # Avoid division by zero
    normals_image = unit_normals.reshape((3, height, width)).transpose(1, 2, 0)

    albedo_image = np.stack(albedos, axis=0).reshape((3, height, width)).transpose(1, 2, 0)
    return albedo_image, normals_image


    










