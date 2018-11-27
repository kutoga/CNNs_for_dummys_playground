from PIL import Image
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Conv2D

smiley = [
    "          ",
    " ##    ## ",
    "          ",
    "          ",
    " ##    ## ",
    "  ######  ",
    "          "
]

# Convert the smiley to a two dimensional numpy array (background=0, forground=1)
smiley_arr = np.asarray([list(map(int, row.replace(' ', '0').replace('#', '1'))) for row in smiley], dtype=np.uint8) * 255

# Allow it to generate images with smileys
def generate_smiley_image(width: int, height: int) -> Image.Image:
    arr = np.zeros((height, width), dtype=np.uint8)
    smiley_shape = smiley_arr.shape
    y0, x0 = [random.randint(0, arr.shape[i] - smiley_shape[i] - 1) for i in range(arr.ndim)]
    arr[y0:(y0 + smiley_shape[0]), x0:(x0 + smiley_shape[1])] = smiley_arr
    return Image.fromarray(arr)

def generate_empty_image(width: int, height: int) -> Image.Image:
    return Image.fromarray(np.zeros((height, width), dtype=np.unit8))

generate_smiley_image(64, 64).save('test.png')

# Trivial: one filter -> it has the size of a smiley
input_image_size = (64, 64)
model = Sequential()
model.add(Conv2D(1, (smiley_arr.shape[1], smiley_arr.shape[0])))
model.add(GlobalMaxPooling())
model.add(Dense(1, 'sigmoid'))

# Add noise to the input image: The filter should still look like the smiley

# More complex: more filters (introduce rotated smileys)
