import argparse
from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# handle command line arguments
parser = argparse.ArgumentParser(description="A script that compresses RGB images using jpeg")

parser.add_argument("filename", type=str, help="The input file path") # can be image or video
parser.add_argument("--quality", "-q", type=int, default=70, help="JPEG Image Quality (0-100)")
parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

args = parser.parse_args()

if not os.path.exists(args.filename):
    print(f"Error: The file '{args.filename}' was not found.")
    sys.exit(1)

if not (0 <= args.quality <= 100):
    print(f"Error: Quality must be between 0 and 100. You provided {args.quality}.")
    sys.exit(1)

if args.verbose:
    print(f"Loading {args.filename}...")
    print(f"Compression set to {args.quality}%")

try:
    # Open the image file
    img = Image.open(args.filename)
    
    # 2. Force conversion to RGB
    # This is crucial. If the user provides a PNG (RGBA) or a B&W image,
    # your math will break later because the dimensions won't be what you expect.
    img = img.convert('RGB')

    # 3. Convert to a NumPy Array
    # This creates a (Height, Width, 3) matrix of integers (0-255)
    image_data = np.array(img)

except Exception as e:
    print(f"Error: Could not process file '{args.filename}'.")
    print(f"Details: {e}")
    sys.exit(1)

# convert from RGB to YCbCr

R = image_data[..., 0]
G = image_data[..., 1]
B = image_data[..., 2]

Y = np.clip(0.299 * R + 0.587 * G + 0.114 * B, 0, 255)
Cb = np.clip(-0.16874 * R - 0.33126 * G + 0.5 * B + 128, 0, 255)
Cr = np.clip(0.5 * R - 0.41869 * G - 0.08131 * B + 128, 0, 255)

fig, axes = plt.subplots(2, 2, figsize=(20, 5))
axes = axes.flatten()

# 1. Show Original RGB
axes[0].imshow(image_data)
axes[0].set_title("Original RGB")
axes[0].axis('off')

# 2. Show Y (Luma) - Should look like a B&W photo
# cmap='gray' tells python to draw low numbers black and high numbers white
axes[1].imshow(Y, cmap='gray')
axes[1].set_title("Y (Brightness)")
axes[1].axis('off')

# compress CbCr channels (4x)
# also called chroma subsampling
# it works because humans suck at perceiving color

Cb_compressed = Cb[::2, ::2]
Cr_compressed = Cr[::2, ::2]

# 3. Show Cb (Blue Difference)
axes[2].imshow(Cb_compressed, cmap='gray')
axes[2].set_title("Cb (Blue Chroma)")
axes[2].axis('off')

# 4. Show Cr (Red Difference)
axes[3].imshow(Cr_compressed, cmap='gray')
axes[3].set_title("Cr (Red Chroma)")
axes[3].axis('off')

plt.tight_layout()
plt.show()

# split image into 8x8 blocks and then apply DCT on these blocks
# by 'image' I mean all channels: Y, Cb, Cr

# take the results of DCT in the form of a matrix and make it sparser
# based on a quality component

# do run-length encoding in a zig-zag on the new matrix

# compress the stream of data further by using Huffman encoding
