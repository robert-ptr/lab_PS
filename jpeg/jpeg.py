import argparse
from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

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

axes[0].imshow(image_data)
axes[0].set_title("Original RGB")
axes[0].axis('off')

axes[1].imshow(Y, cmap='gray')
axes[1].set_title("Y (Brightness)")
axes[1].axis('off')

# compress CbCr channels (4x)
# also called chroma subsampling
# it works because humans suck at perceiving color

Cb_compressed = Cb[::2, ::2]
Cr_compressed = Cr[::2, ::2]

axes[2].imshow(Cb_compressed, cmap='gray')
axes[2].set_title("Cb (Blue Chroma)")
axes[2].axis('off')

axes[3].imshow(Cr_compressed, cmap='gray')
axes[3].set_title("Cr (Red Chroma)")
axes[3].axis('off')

plt.tight_layout()
plt.show()

# split image into 8x8 blocks and then apply DCT on these blocks
# by 'image' I mean all channels: Y, Cb, Cr

def apply_dct_to_channel(channel):
    h, w = channel.shape

    # Add padding, because blocks must be 8x8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
    new_h, new_w = padded.shape

    dct_blocks = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            block = padded[i:i+8, j:j+8]
            block = block - 128.0
            dct_blocks[i:i+8, j:j+8] = dctn(block, norm='ortho')

    return dct_blocks

dct_Y = apply_dct_to_channel(Y)
dct_Cb = apply_dct_to_channel(compressed_Cb)
dct_Cr = apply_dct_to_channel(compressed_Cr)

# take the results of DCT in the form of a matrix and make it sparser
# based on a quality component

# do run-length encoding in a zig-zag on the new matrix

# compress the stream of data further by using Huffman encoding
