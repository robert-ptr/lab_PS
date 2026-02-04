Benchmark Results (Expanded)

| Dataset | Algorithm | Original Size | Compressed Size | Ratio (%) | Time (ms) | Speed (MB/s) |
|---|---|---|---|---|---|---|
| chat_data_for_csharp.txt | Shannon Entropy Limit | 500292 | 315256 | 63.01% | - | - |
| chat_data_for_csharp.txt | Huffman | 500292 | 317488 | 63.46% | 48.10 | 9.92 |
| chat_data_for_csharp.txt | Arithmetic | 500292 | 316062 | 63.18% | 419.77 | 1.14 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255, Lazy=False) + Huffman | 500292 | 256694 | 51.31% | 4915.22 | 0.10 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255, Lazy=False) + Arithmetic | 500292 | 254892 | 50.95% | 5056.63 | 0.09 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255, Lazy=False) | 500292 | 269116 | 53.79% | 4763.44 | 0.10 |
| chat_data_for_csharp.txt | LZSS (W=65535, L=255, Lazy=False) + Huffman | 500292 | 207026 | 41.38% | 24644.89 | 0.02 |
| chat_data_for_csharp.txt | LZSS (W=65535, L=255, Lazy=False) + Arithmetic | 500292 | 204942 | 40.96% | 24484.36 | 0.02 |
| chat_data_for_csharp.txt | LZSS (W=65535, L=255, Lazy=False) | 500292 | 210688 | 42.11% | 24490.84 | 0.02 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255, Lazy=True) + Huffman | 500292 | 247597 | 49.49% | 8849.61 | 0.05 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255, Lazy=True) + Arithmetic | 500292 | 245638 | 49.10% | 8969.39 | 0.05 |
| chat_data_for_csharp.txt | LZSS (W=8192, L=255, Lazy=True) | 500292 | 258053 | 51.58% | 8651.00 | 0.06 |
| chat_data_for_csharp.txt | LZSS (W=65535, L=255, Lazy=True) + Huffman | 500292 | 198768 | 39.73% | 50214.12 | 0.01 |
| chat_data_for_csharp.txt | LZSS (W=65535, L=255, Lazy=True) + Arithmetic | 500292 | 196767 | 39.33% | 49855.37 | 0.01 |
| chat_data_for_csharp.txt | LZSS (W=65535, L=255, Lazy=True) | 500292 | 201320 | 40.24% | 49477.67 | 0.01 |
| chat_data_for_csharp.txt | BWT+MTF + Huffman | 500292 | 209365 | 41.85% | 320.88 | 1.49 |
| chat_data_for_csharp.txt | BWT+MTF + Arithmetic | 500292 | 208594 | 41.69% | 670.53 | 0.71 |
| twitch_chat_raw.txt | Shannon Entropy Limit | 512101 | 339351 | 66.27% | - | - |
| twitch_chat_raw.txt | Huffman | 512101 | 342074 | 66.80% | 43.44 | 11.24 |
| twitch_chat_raw.txt | Arithmetic | 512101 | 334582 | 65.34% | 435.74 | 1.12 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255, Lazy=False) + Huffman | 512101 | 261096 | 50.99% | 5700.36 | 0.09 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255, Lazy=False) + Arithmetic | 512101 | 258970 | 50.57% | 5909.83 | 0.08 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255, Lazy=False) | 512101 | 273061 | 53.32% | 5709.41 | 0.09 |
| twitch_chat_raw.txt | LZSS (W=65535, L=255, Lazy=False) + Huffman | 512101 | 208230 | 40.66% | 26964.45 | 0.02 |
| twitch_chat_raw.txt | LZSS (W=65535, L=255, Lazy=False) + Arithmetic | 512101 | 206214 | 40.27% | 26931.63 | 0.02 |
| twitch_chat_raw.txt | LZSS (W=65535, L=255, Lazy=False) | 512101 | 212709 | 41.54% | 26721.30 | 0.02 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255, Lazy=True) + Huffman | 512101 | 255042 | 49.80% | 9017.26 | 0.05 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255, Lazy=True) + Arithmetic | 512101 | 252981 | 49.40% | 9191.10 | 0.05 |
| twitch_chat_raw.txt | LZSS (W=8192, L=255, Lazy=True) | 512101 | 266221 | 51.99% | 9165.94 | 0.05 |
| twitch_chat_raw.txt | LZSS (W=65535, L=255, Lazy=True) + Huffman | 512101 | 200740 | 39.20% | 49300.68 | 0.01 |
| twitch_chat_raw.txt | LZSS (W=65535, L=255, Lazy=True) + Arithmetic | 512101 | 198745 | 38.81% | 48632.11 | 0.01 |
| twitch_chat_raw.txt | LZSS (W=65535, L=255, Lazy=True) | 512101 | 204398 | 39.91% | 48348.33 | 0.01 |
| twitch_chat_raw.txt | BWT+MTF + Huffman | 512101 | 240601 | 46.98% | 344.96 | 1.42 |
| twitch_chat_raw.txt | BWT+MTF + Arithmetic | 512101 | 237448 | 46.37% | 720.31 | 0.68 |
| esa_telemetry.bin | Shannon Entropy Limit | 1048576 | 265969 | 25.36% | - | - |
| esa_telemetry.bin | Huffman | 1048576 | 297978 | 28.42% | 34.45 | 29.02 |
| esa_telemetry.bin | Arithmetic | 1048576 | 269178 | 25.67% | 800.78 | 1.25 |
| esa_telemetry.bin | LZSS (W=8192, L=255, Lazy=False) + Huffman | 1048576 | 10500 | 1.00% | 125.54 | 7.97 |
| esa_telemetry.bin | LZSS (W=8192, L=255, Lazy=False) + Arithmetic | 1048576 | 9356 | 0.89% | 137.84 | 7.25 |
| esa_telemetry.bin | LZSS (W=8192, L=255, Lazy=False) | 1048576 | 14949 | 1.43% | 124.88 | 8.01 |
| esa_telemetry.bin | LZSS (W=65535, L=255, Lazy=False) + Huffman | 1048576 | 8258 | 0.79% | 930.17 | 1.08 |
| esa_telemetry.bin | LZSS (W=65535, L=255, Lazy=False) + Arithmetic | 1048576 | 7086 | 0.68% | 976.40 | 1.02 |
| esa_telemetry.bin | LZSS (W=65535, L=255, Lazy=False) | 1048576 | 14889 | 1.42% | 927.60 | 1.08 |
| esa_telemetry.bin | LZSS (W=8192, L=255, Lazy=True) + Huffman | 1048576 | 10500 | 1.00% | 177.94 | 5.62 |
| esa_telemetry.bin | LZSS (W=8192, L=255, Lazy=True) + Arithmetic | 1048576 | 9356 | 0.89% | 190.67 | 5.24 |
| esa_telemetry.bin | LZSS (W=8192, L=255, Lazy=True) | 1048576 | 14949 | 1.43% | 187.14 | 5.34 |
| esa_telemetry.bin | LZSS (W=65535, L=255, Lazy=True) + Huffman | 1048576 | 8256 | 0.79% | 1292.53 | 0.77 |
| esa_telemetry.bin | LZSS (W=65535, L=255, Lazy=True) + Arithmetic | 1048576 | 7084 | 0.68% | 1293.43 | 0.77 |
| esa_telemetry.bin | LZSS (W=65535, L=255, Lazy=True) | 1048576 | 14887 | 1.42% | 1278.00 | 0.78 |
| esa_telemetry.bin | BWT+MTF + Huffman | 1048576 | 134809 | 12.86% | 208042.08 | 0.00 |
| esa_telemetry.bin | BWT+MTF + Arithmetic | 1048576 | 9460 | 0.90% | 209112.04 | 0.00 |
| minecraft_dataset.bin | Shannon Entropy Limit | 1048576 | 505432 | 48.20% | - | - |
| minecraft_dataset.bin | Huffman | 1048576 | 509491 | 48.59% | 54.56 | 18.33 |
| minecraft_dataset.bin | Arithmetic | 1048576 | 442923 | 42.24% | 808.22 | 1.24 |
| minecraft_dataset.bin | LZSS (W=8192, L=255, Lazy=False) + Huffman | 1048576 | 234507 | 22.36% | 7996.09 | 0.13 |
| minecraft_dataset.bin | LZSS (W=8192, L=255, Lazy=False) + Arithmetic | 1048576 | 232129 | 22.14% | 8228.01 | 0.12 |
| minecraft_dataset.bin | LZSS (W=8192, L=255, Lazy=False) | 1048576 | 266919 | 25.46% | 7932.63 | 0.13 |
| minecraft_dataset.bin | LZSS (W=65535, L=255, Lazy=False) + Huffman | 1048576 | 231310 | 22.06% | 46335.14 | 0.02 |
| minecraft_dataset.bin | LZSS (W=65535, L=255, Lazy=False) + Arithmetic | 1048576 | 229099 | 21.85% | 46539.13 | 0.02 |
| minecraft_dataset.bin | LZSS (W=65535, L=255, Lazy=False) | 1048576 | 249933 | 23.84% | 46408.85 | 0.02 |
| minecraft_dataset.bin | LZSS (W=8192, L=255, Lazy=True) + Huffman | 1048576 | 223244 | 21.29% | 15355.53 | 0.07 |
| minecraft_dataset.bin | LZSS (W=8192, L=255, Lazy=True) + Arithmetic | 1048576 | 220911 | 21.07% | 15733.04 | 0.06 |
| minecraft_dataset.bin | LZSS (W=8192, L=255, Lazy=True) | 1048576 | 254111 | 24.23% | 15370.40 | 0.07 |
| minecraft_dataset.bin | LZSS (W=65535, L=255, Lazy=True) + Huffman | 1048576 | 217831 | 20.77% | 95616.50 | 0.01 |
| minecraft_dataset.bin | LZSS (W=65535, L=255, Lazy=True) + Arithmetic | 1048576 | 215652 | 20.57% | 95362.79 | 0.01 |
| minecraft_dataset.bin | LZSS (W=65535, L=255, Lazy=True) | 1048576 | 236309 | 22.54% | 95246.38 | 0.01 |
| minecraft_dataset.bin | BWT+MTF + Huffman | 1048576 | 251416 | 23.98% | 18745.95 | 0.05 |
| minecraft_dataset.bin | BWT+MTF + Arithmetic | 1048576 | 206529 | 19.70% | 19466.81 | 0.05 |
