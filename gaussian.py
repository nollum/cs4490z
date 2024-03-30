import os
import sys
import numpy as np
from skimage import io, filters, img_as_ubyte

def reduce_sharpness(image_path, amount):
    # Read the image
    image = io.imread(image_path)
    
    # Reduce sharpness
    blurred_image = filters.gaussian(image, sigma=amount)
    
    return blurred_image

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_directory> <output_directory> <amount>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    amount = float(sys.argv[3])

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    
    for f in files:
        input_path = os.path.join(input_directory, f)
        output_path = os.path.join(output_directory, f)
        blurred_image = reduce_sharpness(input_path, amount)
        # Convert the image array to the correct data type
        blurred_image = img_as_ubyte(blurred_image)
        io.imsave(output_path, blurred_image)
