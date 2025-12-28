import argparse
import PIL
import os
import sys

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

# convert from RGB to YCbCr


# compress CbCr channels (4x)
# also called chroma subsampling
# it works because humans suck at perceiving color

# split image into 8x8 blocks and then apply DCT on these blocks
# by 'image' I mean all channels: Y, Cb, Cr

# take the results of DCT in the form of a matrix and make it sparser
# based on a quality component

# do run-length encoding in a zig-zag on the new matrix

# compress the stream of data further by using Huffman encoding
