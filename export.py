#!/usr/bin/env python3
import os
import sys
import struct
import zipfile
import pickle
import io
import json
import numpy as np

MAGIC = b"PIMI"
VERSION = 1

# Maps checkpoint name -> (canonical name, should_transpose)
# In PyTorch, Linear weight is [out_features, in_features].
# In pimi, contract is C = A x B where B is [in_features, out_features].
# Therefore, all 2D Linear projection weights must be transposed.
NAME_MAP = {
    "transformer.wte.weight": ("embed.weight", False),         # [50257, 768] (row lookup)
    "transformer.wpe.weight": ("pos_embed.weight", False),     # [2048, 768]  (row lookup)
    "transformer.ln_f.weight": ("ln_f.weight", False),         # [768]
    "transformer.ln_f.bias": ("ln_f.bias", False),             # [768]
}

for i in range(4):
    prefix = f"transformer.h.{i}."
    target = f"layers.{i}."
    NAME_MAP.update({
        f"{prefix}ln_1.weight": (f"{target}ln_1.weight", False),
        f"{prefix}ln_1.bias": (f"{target}ln_1.bias", False),
        f"{prefix}attn.attention.q_proj.weight": (f"{target}attn.q_proj.weight", True),
        f"{prefix}attn.attention.k_proj.weight": (f"{target}attn.k_proj.weight", True),
        f"{prefix}attn.attention.v_proj.weight": (f"{target}attn.v_proj.weight", True),
        f"{prefix}attn.attention.out_proj.weight": (f"{target}attn.out_proj.weight", True),
        f"{prefix}attn.attention.out_proj.bias": (f"{target}attn.out_proj.bias", False),
        f"{prefix}ln_2.weight": (f"{target}ln_2.weight", False),
        f"{prefix}ln_2.bias": (f"{target}ln_2.bias", False),
        f"{prefix}mlp.c_fc.weight": (f"{target}mlp.fc.weight", True),     # [3072, 768] -> [768, 3072]
        f"{prefix}mlp.c_fc.bias": (f"{target}mlp.fc.bias", False),
        f"{prefix}mlp.c_proj.weight": (f"{target}mlp.proj.weight", True), # [768, 3072] -> [3072, 768]
        f"{prefix}mlp.c_proj.bias": (f"{target}mlp.proj.bias", False),
    })

class FakeTensor:
    def __init__(self, storage, storage_offset, size, stride, *args):
        self.storage = storage
        self.size = size

class FakeStorage:
    def __init__(self, *args): pass

class PyTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2": return FakeTensor
        if module == "torch" and "Storage" in name: return FakeStorage
        if module == "collections" and name == "OrderedDict": return dict
        return lambda *args, **kwargs: None

    def persistent_load(self, pid): return pid

def export_model(model_bin_path, config_path, output_pimi_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    vocab_size = cfg.get("vocab_size", 50257)
    hidden_dim = cfg.get("hidden_size", 768)
    num_layers = cfg.get("num_layers", 4)
    num_heads = cfg.get("num_heads", 16)
    max_seq_len = cfg.get("max_position_embeddings", 2048)

    with zipfile.ZipFile(model_bin_path, "r") as z:
        pkl_bytes = z.read("pytorch_model/data.pkl")
        unpickler = PyTorchUnpickler(io.BytesIO(pkl_bytes))
        state_dict = unpickler.load()

        export_list = []
        for orig_name, (canon_name, should_transpose) in NAME_MAP.items():
            if orig_name not in state_dict:
                print(f"warning: missing tensor {orig_name}")
                continue
            tensor_info = state_dict[orig_name]
            storage_key = tensor_info.storage[2]
            archive_file = f"pytorch_model/data/{storage_key}"
            raw_bytes = z.read(archive_file)

            shape = list(tensor_info.size)
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
            arr = arr.reshape(shape)

            if should_transpose:
                # Transpose 2D matrix from [out, in] to [in, out]
                arr = np.ascontiguousarray(arr.T)
                shape = list(arr.shape)

            raw_out = arr.tobytes()
            export_list.append({
                "canon_name": canon_name,
                "shape": shape,
                "numel": arr.size,
                "data": raw_out
            })

        # Add lm_head.weight transposed from wte [50257, 768] -> [768, 50257]
        if "transformer.wte.weight" in state_dict:
            wte_info = state_dict["transformer.wte.weight"]
            wte_key = wte_info.storage[2]
            wte_bytes = z.read(f"pytorch_model/data/{wte_key}")
            wte_arr = np.frombuffer(wte_bytes, dtype=np.float32).reshape(list(wte_info.size))
            lm_head_arr = np.ascontiguousarray(wte_arr.T)
            export_list.append({
                "canon_name": "lm_head.weight",
                "shape": list(lm_head_arr.shape),
                "numel": lm_head_arr.size,
                "data": lm_head_arr.tobytes()
            })

    header_size = 64
    entry_size = 128
    num_tensors = len(export_list)
    table_size = num_tensors * entry_size
    current_data_offset = (header_size + table_size + 63) & ~63
    total_weights_bytes = sum(len(item["data"]) for item in export_list)

    print(f"exporting {num_tensors} tensors, raw float32 size: {total_weights_bytes / (1024*1024):.2f} MB")

    with open(output_pimi_path, "wb") as out:
        header = struct.pack(
            "<4sIIQIIIII24s",
            MAGIC, VERSION, num_tensors, total_weights_bytes,
            vocab_size, hidden_dim, num_layers, num_heads, max_seq_len,
            b"\x00" * 24
        )
        out.write(header)

        aligned_data_offset = current_data_offset
        entries = []
        for item in export_list:
            dims4 = [1, 1, 1, 1]
            for i, d in enumerate(item["shape"]): dims4[i] = d
            name_bytes = item["canon_name"].encode("utf-8")[:63]
            name_buf = name_bytes + b"\x00" * (64 - len(name_bytes))
            nbytes = len(item["data"])
            entry = struct.pack(
                "<64si4iQQQ5I",
                name_buf, len(item["shape"]),
                dims4[0], dims4[1], dims4[2], dims4[3],
                aligned_data_offset, item["numel"], nbytes,
                0, 0, 0, 0, 0
            )
            entries.append((entry, item["data"]))
            aligned_data_offset += (nbytes + 63) & ~63

        for entry, _ in entries: out.write(entry)
        pos = out.tell()
        if pos < current_data_offset: out.write(b"\x00" * (current_data_offset - pos))
        for _, data in entries:
            out.write(data)
            pad = ((len(data) + 63) & ~63) - len(data)
            if pad > 0: out.write(b"\x00" * pad)

    final_size = os.path.getsize(output_pimi_path)
    print(f"saved: {output_pimi_path} ({final_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    export_model("models/pytorch_model.bin", "models/config.json", "models/tinystories_33m.pimi")
    if os.path.exists("models/vocab.json") and os.path.exists("models/merges.txt"):
        from export_tokenizer import export_tokenizer
        export_tokenizer("models/vocab.json", "models/merges.txt", "models/tokenizer.bin")

