# pimi

running an LLM on a $5 nvidia gpu from 2010.

i had an old quadro 2000 (fermi, compute capability 2.1, 1gb vram) sitting in a drawer. modern cuda doesn't support it, and installing a 10-year-old cuda 8.0 toolkit sounded painful.

so i wrote this: a minimal C runtime that talks straight to `nvcuda.dll` via the cuda driver api, hands raw PTX to the driver's built-in JIT compiler, and runs [TinyStories-Instruct-33M](https://huggingface.co/roneneldan/TinyStories-Instruct-33M) (GPT-Neo 4L/768H) directly on the GPU.

no pytorch at runtime. no cuda toolkit. no nvcc. no dependencies beyond msvc and windows.
runs at ~8 tokens/sec on hardware released when Obama was in his first term.

---

```text
loading tokenizer: models/tokenizer.bin
loading model: models/tinystories_33m.pimi
pimi: initialized on Quadro 2000 (sm_21), 817 mb free

--- generation ---
Words: dog, ball, park
Story: Once upon a time, there was a little boy named Timmy. He loved to play with his red ball in the park every day. One day while he was playing with his ball he saw some other kids playing too. They were all having fun and laughing together. 

Timmy wanted
--- done ---

generated 60 tokens in 12349.8 ms (4.86 tok/s)
```

---

### how it works

- **driver api + ptx JIT**: loads `C:\Windows\System32\nvcuda.dll` at runtime. driver compiles PTX (`sm_20`) on the fly. no cuda SDK needed.
- **hybrid placement**: weights and activation loop live 100% on VRAM (~435 MB total).
- **pcie avoidance**: only slices and projects the last hidden state $[1, 768] \times [768, 50257]$ through LM head. copies 200 KB to host for CPU sampling instead of dumping 20 MB across PCIe every step.
- **bpe tokenizer**: pure C with an open-addressing hash table and custom scanner. zero external regex libraries, loads in 1 ms, exact match with HF tokenizer.
- **fun bug**: gpt-neo self-attention does *not* scale by $\sqrt{d_{head}}$ ($scores = Q K^T$). took me two hours to realize why it was repeating words until i checked `GPTNeoSelfAttention` in transformers.

### quickstart

#### 1. build
Open "x64 Native Tools Command Prompt for VS 2022" (or run `vcvars64.bat`):
```bat
build.bat
```

#### 2. download weights & export
Download `pytorch_model.bin`, `config.json`, `vocab.json`, `merges.txt` from [roneneldan/TinyStories-Instruct-33M](https://huggingface.co/roneneldan/TinyStories-Instruct-33M) into `models/`:
```bash
python export.py
```
This converts the checkpoint to a flat `.pimi` binary and exports `tokenizer.bin`.

#### 3. run
```bat
pimi.exe "Once upon a time"
```

or instruct prompt:
```bat
pimi.exe -n 60 --temp 0.6 --top-p 0.9 --repeat 1.15 "Words: dog, ball, park\nStory:"
```

### cli flags

```text
-n <int>         max new tokens (default: 64)
--temp <float>   temperature (default: 0.7, set 0.0 for greedy)
--top-p <float>  top-p sampling (default: 0.9)
--repeat <float> repetition penalty (default: 1.1)
--seed <int>     random seed (default: 42)
-m <path>        model file (default: models/tinystories_33m.pimi)
-t <path>        tokenizer file (default: models/tokenizer.bin)
```

### repo structure

```text
pimi/
├── include/
│   ├── pimi.h            # tensor & ops api
│   ├── model.h           # gpt-neo model api
│   └── tokenizer.h       # bpe tokenizer api
├── src/
│   ├── cuda_drv.c/.h     # nvcuda.dll dynamic loader
│   ├── runtime.c         # context & ptx JIT manager
│   ├── kernels.ptx       # hand-written / minimal ptx kernels
│   ├── tensor.c          # tensor allocator & transfers
│   ├── ops.c             # matmul, layernorm, gelu, attention
│   ├── model.c           # forward pass & layer pipeline
│   ├── tokenizer.c       # c bpe tokenizer
│   └── main.c            # cli entrypoint
├── export.py             # converts pytorch checkpoint to .pimi
├── export_tokenizer.py   # standalone tokenizer exporter
└── build.bat             # one-click build
```

### license
MIT
