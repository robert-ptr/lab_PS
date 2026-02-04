# Data Compression Benchmark Results
Date: 2/3/2026 12:35:55 AM

| Dataset | Algorithm | Original Size | Compressed Size | Ratio (%) | Time (ms) | Speed (MB/s) |
|---|---|---|---|---|---|---|
| chat_data_for_csharp.txt | Huffman | 500292 | 317488 | 63.46% | 20.41 | 23.38 |
| chat_data_for_csharp.txt | LZSS (W=4096, L=255) + Huffman | 500292 | 274884 | 54.94% | 454.05 | 1.05 |
| chat_data_for_csharp.txt | LZSS (W=4096, L=255) + Arithmetic | 500292 | 273080 | 54.58% | 431.83 | 1.10 |
| chat_data_for_csharp.txt | LZSS (W=1024, L=64) + Huffman | 500292 | 313561 | 62.68% | 156.90 | 3.04 |
| chat_data_for_csharp.txt | LZSS (W=1024, L=64) + Arithmetic | 500292 | 311332 | 62.23% | 217.50 | 2.19 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255) + Huffman | 500292 | 256694 | 51.31% | 583.71 | 0.82 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255) + Arithmetic | 500292 | 254892 | 50.95% | 634.12 | 0.75 |
| chat_data_for_csharp.txt | BWT+MTF + Huffman | 500292 | 209365 | 41.85% | 329.92 | 1.45 |
| chat_data_for_csharp.txt | BWT+MTF + Arithmetic | 500292 | 208594 | 41.69% | 352.46 | 1.35 |
| twitch_chat_raw.txt | Huffman | 512101 | 342074 | 66.80% | 10.52 | 46.44 |
| twitch_chat_raw.txt | LZSS (W=4096, L=255) + Huffman | 512101 | 283056 | 55.27% | 419.57 | 1.16 |
| twitch_chat_raw.txt | LZSS (W=4096, L=255) + Arithmetic | 512101 | 281089 | 54.89% | 467.02 | 1.05 |
| twitch_chat_raw.txt | LZSS (W=1024, L=64) + Huffman | 512101 | 329758 | 64.39% | 164.83 | 2.96 |
| twitch_chat_raw.txt | LZSS (W=1024, L=64) + Arithmetic | 512101 | 327518 | 63.96% | 236.85 | 2.06 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255) + Huffman | 512101 | 261096 | 50.99% | 709.73 | 0.69 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255) + Arithmetic | 512101 | 258970 | 50.57% | 741.37 | 0.66 |
| twitch_chat_raw.txt | BWT+MTF + Huffman | 512101 | 240601 | 46.98% | 285.96 | 1.71 |
| twitch_chat_raw.txt | BWT+MTF + Arithmetic | 512101 | 237448 | 46.37% | 378.29 | 1.29 |
| esa_telemetry.bin | Huffman | 1048576 | 297978 | 28.42% | 6.45 | 154.95 |
| esa_telemetry.bin | LZSS (W=4096, L=255) + Huffman | 1048576 | 10731 | 1.02% | 8.61 | 116.08 |
| esa_telemetry.bin | LZSS (W=4096, L=255) + Arithmetic | 1048576 | 9557 | 0.91% | 11.41 | 87.68 |
| esa_telemetry.bin | LZSS (W=1024, L=64) + Huffman | 1048576 | 26458 | 2.52% | 4.08 | 245.23 |
| esa_telemetry.bin | LZSS (W=1024, L=64) + Arithmetic | 1048576 | 25229 | 2.41% | 14.63 | 68.36 |
| esa_telemetry.bin | LZSS (W=8192, L=255) + Huffman | 1048576 | 10500 | 1.00% | 22.61 | 44.22 |
| esa_telemetry.bin | LZSS (W=8192, L=255) + Arithmetic | 1048576 | 9356 | 0.89% | 17.57 | 56.91 |
| esa_telemetry.bin | BWT+MTF + Huffman | 1048576 | 134809 | 12.86% | 206618.04 | 0.00 |
| esa_telemetry.bin | BWT+MTF + Arithmetic | 1048576 | 9460 | 0.90% | 206841.72 | 0.00 |
| minecraft_dataset.bin | Huffman | 1048576 | 509491 | 48.59% | 14.33 | 69.81 |
| minecraft_dataset.bin | LZSS (W=4096, L=255) + Huffman | 1048576 | 235648 | 22.47% | 628.45 | 1.59 |
| minecraft_dataset.bin | LZSS (W=4096, L=255) + Arithmetic | 1048576 | 233082 | 22.23% | 672.03 | 1.49 |
| minecraft_dataset.bin | LZSS (W=1024, L=64) + Huffman | 1048576 | 249584 | 23.80% | 207.84 | 4.81 |
| minecraft_dataset.bin | LZSS (W=1024, L=64) + Arithmetic | 1048576 | 246379 | 23.50% | 235.44 | 4.25 |
| minecraft_dataset.bin | LZSS (W=8192, L=255) + Huffman | 1048576 | 234507 | 22.36% | 930.17 | 1.08 |
| minecraft_dataset.bin | LZSS (W=8192, L=255) + Arithmetic | 1048576 | 232129 | 22.14% | 989.32 | 1.01 |
| minecraft_dataset.bin | BWT+MTF + Huffman | 1048576 | 251416 | 23.98% | 18529.31 | 0.05 |
| minecraft_dataset.bin | BWT+MTF + Arithmetic | 1048576 | 206529 | 19.70% | 18657.03 | 0.05 |
