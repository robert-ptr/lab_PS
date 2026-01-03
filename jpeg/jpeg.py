import argparse
from PIL import Image
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

# Standard Quantization Table for Y 
Tb_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Standard Quantization Table for Chrominence
Tb_Chroma = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

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
    img = Image.open(args.filename)
    img = img.convert('RGB')
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

# fig, axes = plt.subplots(2, 2, figsize=(20, 5))
# axes = axes.flatten()

# axes[0].imshow(image_data)
# axes[0].set_title("Original RGB")
# axes[0].axis('off')

# axes[1].imshow(Y, cmap='gray')
# axes[1].set_title("Y (Brightness)")
# axes[1].axis('off')

# compress CbCr channels (4x)
# also called chroma subsampling
# it works because humans suck at perceiving color

Cb_compressed = Cb[::2, ::2]
Cr_compressed = Cr[::2, ::2]

# axes[2].imshow(Cb_compressed, cmap='gray')
# axes[2].set_title("Cb (Blue Chroma)")
# axes[2].axis('off')

# axes[3].imshow(Cr_compressed, cmap='gray')
# axes[3].set_title("Cr (Red Chroma)")
# axes[3].axis('off')

# plt.tight_layout()
# plt.show()

# split image into 8x8 blocks and then apply DCT on these blocks
# by 'image' I mean all channels: Y, Cb, Cr

def apply_dct_to_channel(channel):
    h, w = channel.shape

    # Add padding, because blocks must be 8x8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
    new_h, new_w = padded.shape

    blocks = (padded.reshape(new_h // 8, 8, new_w // 8, 8)
                    .transpose(0, 2, 1, 3)
                    .reshape(-1, 8, 8))

    blocks = blocks - 128.0

    dct_blocks = dctn(blocks, axes=(1, 2), norm='ortho')

    return dct_blocks

dct_Y = apply_dct_to_channel(Y)
dct_Cb = apply_dct_to_channel(Cb_compressed)
dct_Cr = apply_dct_to_channel(Cr_compressed)

# take the results of DCT in the form of a matrix and make it sparser
# based on a quality component

if args.quality < 50:
    S = 5000 / args.quality
else:
    S = 200 - 2 * args.quality

Ts_Y = np.floor((S * Tb_Y + 50) / 100)
Ts_Y = np.clip(Ts_Y, 1, 255)
Ts_Y = Ts_Y.astype(np.int32)

Ts_Chroma = np.floor((S * Tb_Chroma + 50) / 100)
Ts_Chroma = np.clip(Ts_Chroma, 1, 255)
Ts_Chroma = Ts_Chroma.astype(np.int32)

quantized_Y = np.round(dct_Y / Ts_Y)
quantized_Cb = np.round(dct_Cb / Ts_Chroma)
quantized_Cr = np.round(dct_Cr / Ts_Chroma)

# do run-length encoding in a zig-zag on the new matrix
def do_the_zig_zag(block):
    flipped = np.flipud(block)

    return np.concatenate([
        np.diagonal(flipped, offset=i)[::1 if i % 2 == 0 else -1]
        for i in range(-7, 8)
    ])

def flatten_blocks(quantized_channel):
    flattened_blocks = []

    for block in quantized_channel:
        flat_block = do_the_zig_zag(block)
            
        flattened_blocks.append(flat_block)
            
    return flattened_blocks

flattened_Y = flatten_blocks(quantized_Y)
flattened_Cb = flatten_blocks(quantized_Cb)
flattened_Cr = flatten_blocks(quantized_Cr)

def RLE_encode(block_stream):
    encoded = []
    
    encoded.append(block_stream[0]) 
    
    zeros = 0
    for i in range(1, 64):
        val = block_stream[i]
        
        if val == 0:
            zeros += 1
        else:
            encoded.append((zeros, val))
            zeros = 0
            
    if zeros > 0:
        encoded.append((0, 0))
        
    return encoded

if args.verbose:
    print("Performing Run-Length Encoding...")

# apply RLE to all channels
encoded_Y = [RLE_encode(block) for block in flattened_Y]
encoded_Cb = [RLE_encode(block) for block in flattened_Cb]
encoded_Cr = [RLE_encode(block) for block in flattened_Cr]

# compress the stream of data further by using Huffman encoding

# De-Quantize
Y_recon_blocks = quantized_Y * Ts_Y
Cb_recon_blocks = quantized_Cb * Ts_Chroma
Cr_recon_blocks = quantized_Cr * Ts_Chroma

# IDCT
Y_idct = idctn(Y_recon_blocks, axes=(1, 2), norm='ortho')
Cb_idct = idctn(Cb_recon_blocks, axes=(1, 2), norm='ortho')
Cr_idct = idctn(Cr_recon_blocks, axes=(1, 2), norm='ortho')

Y_idct += 128
Cb_idct += 128
Cr_idct += 128

def merge_blocks(blocks, h, w):
    return (blocks.reshape(h // 8, w // 8, 8, 8)
                  .transpose(0, 2, 1, 3)
                  .reshape(h, w))

h_orig, w_orig = image_data.shape[:2]

h_y_padded = int(np.ceil(h_orig / 8) * 8)
w_y_padded = int(np.ceil(w_orig / 8) * 8)

h_chroma = int(np.ceil(h_orig / 2)) # subsampled size
w_chroma = int(np.ceil(w_orig / 2)) 

h_chroma_padded = int(np.ceil(h_chroma / 8) * 8) # padded size
w_chroma_padded = int(np.ceil(w_chroma / 8) * 8)

Y_full = merge_blocks(Y_idct, h_y_padded, w_y_padded)
Cb_full = merge_blocks(Cb_idct, h_chroma_padded, w_chroma_padded)
Cr_full = merge_blocks(Cr_idct, h_chroma_padded, w_chroma_padded)

# upsample Cb/Cr
Cb_up = Cb_full.repeat(2, axis=0).repeat(2, axis=1)
Cr_up = Cr_full.repeat(2, axis=0).repeat(2, axis=1)

# crop to original image size
Y_final = Y_full[:h_orig, :w_orig]
Cb_final = Cb_up[:h_orig, :w_orig]
Cr_final = Cr_up[:h_orig, :w_orig]

# convert YCbCr to RGB
R_out = Y_final + 1.402 * (Cr_final - 128)
G_out = Y_final - 0.344136 * (Cb_final - 128) - 0.714136 * (Cr_final - 128)
B_out = Y_final + 1.772 * (Cb_final - 128)

img_reconstructed = np.dstack((R_out, G_out, B_out))
img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)

output_filename = f"compressed_q{args.quality}.jpg"
Image.fromarray(img_reconstructed).save(output_filename)
print(f"Saved reconstructed image to: {output_filename}")
