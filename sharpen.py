import os
import sys
import numpy as np
from skimage.filters import unsharp_mask, gaussian
from skimage import color, io, img_as_ubyte

def sharpen_image(image_path, amount, radius, sigma):
    image = io.imread(image_path)
    
    image_gray = color.rgb2gray(image)
    
    # Denoise the image using a Gaussian filter
    denoised_image = gaussian(image_gray, sigma=sigma)
    
    unsharp_image = unsharp_mask(denoised_image, radius=radius, amount=amount)
    
    sharpened_image = img_as_ubyte(unsharp_image)
    
    return sharpened_image

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py <input_directory> <output_directory> <amount> <radius> <sigma>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    amount = float(sys.argv[3])
    radius = float(sys.argv[4])
    sigma = float(sys.argv[5])

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    
    for f in files:
        input_path = os.path.join(input_directory, f)
        output_path = os.path.join(output_directory, f)
        sharpened_image = sharpen_image(input_path, amount, radius, sigma)
        io.imsave(output_path, sharpened_image)
