import os
import sys
import numpy as np
from skimage import color, io, img_as_ubyte
from skimage.util import random_noise

def add_noise_to_image(image_path, noise_amount):
    image = io.imread(image_path)
    
    noisy_image = random_noise(image, var=noise_amount**2)
    
    noisy_image = img_as_ubyte(noisy_image)
    
    return noisy_image

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_directory> <output_directory> <noise_amount>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    noise_amount = float(sys.argv[3])

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    
    for f in files:
        input_path = os.path.join(input_directory, f)
        output_path = os.path.join(output_directory, f)
        noisy_image = add_noise_to_image(input_path, noise_amount)
        io.imsave(output_path, noisy_image)