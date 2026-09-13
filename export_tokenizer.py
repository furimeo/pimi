#!/usr/bin/env python3
import json
import struct
import os

MAGIC = b"PTOK"
VERSION = 1

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('\xa1'), ord('\xac')+1)) + list(range(ord('\xae'), ord('\xff')+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(i) for i in cs]))

def export_tokenizer(vocab_path, merges_path, out_path):
    print(f"Exporting tokenizer from {vocab_path} and {merges_path} -> {out_path}")
    
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(merges_path, "r", encoding="utf-8") as f:
        merges_lines = [line.strip().split() for line in f.read().splitlines()[1:50001] if line.strip()]

    vocab_size = len(vocab)
    num_merges = len(merges_lines)

    # 1. Byte to token ID mapping (0..255)
    byte_to_tok = [vocab[b2u[b]] for b in range(256)]

    # 2. Merges table: id_a, id_b, id_merged, rank
    merges_data = []
    for rank, (a, b) in enumerate(merges_lines):
        id_a = vocab[a]
        id_b = vocab[b]
        id_merged = vocab[a + b]
        merges_data.append((id_a, id_b, id_merged, rank))

    # 3. Invert vocab to get raw UTF-8 string for each token ID
    id_to_bytes = {}
    for word, idx in vocab.items():
        id_to_bytes[idx] = bytes([u2b[c] for c in word])

    with open(out_path, "wb") as f:
        # Header: magic(4), version(4), vocab_size(4), num_merges(4)
        f.write(struct.pack("<4sIII", MAGIC, VERSION, vocab_size, num_merges))

        # Byte to token table: 256 * uint32
        f.write(struct.pack(f"<{256}I", *byte_to_tok))

        # Merges table: num_merges * 4 uint32 (id_a, id_b, id_merged, rank)
        for entry in merges_data:
            f.write(struct.pack("<IIII", *entry))

        # Vocab strings: for each id in 0..vocab_size-1: len(uint32), string bytes
        for i in range(vocab_size):
            raw = id_to_bytes[i]
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported {out_path}: {size_mb:.2f} MB ({vocab_size} tokens, {num_merges} merges)")

if __name__ == "__main__":
    vocab_p = "models/vocab.json"
    merges_p = "models/merges.txt"
    out_p = "models/tokenizer.bin"
    export_tokenizer(vocab_p, merges_p, out_p)
