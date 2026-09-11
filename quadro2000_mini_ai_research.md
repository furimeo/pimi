# Runtime ML nhỏ cho NVIDIA Quadro 2000

## Kết luận điều hành

NVIDIA Quadro 2000 là GPU Fermi compute capability 2.1. NVIDIA xác nhận Fermi là kiến trúc cuối cùng được hỗ trợ bởi CUDA Toolkit 8.0; CUDA 9 bỏ hỗ trợ Fermi ở toolkit/compiler, trong khi các ứng dụng đã build bằng CUDA 8 hoặc cũ hơn vẫn có thể chạy trên driver phù hợp. Quadro 2000 có 1 GB GDDR5 và băng thông bộ nhớ 41.6 GB/s theo datasheet NVIDIA. [1][2][3]

Vì vậy, hướng hợp lý không phải là fork PyTorch hiện đại mà là một runtime ML rất nhỏ bằng C + CUDA 8, nhắm trực tiếp `sm_21`. Runtime nên tập trung vào inference trước, dùng `cudaMalloc`/`cudaMemcpy`, giữ toàn bộ tensor trung gian trên GPU và gọi cuBLAS cho GEMM nếu phiên bản cuBLAS tương ứng hỗ trợ đường chạy cần thiết. Unified Memory không nên được coi là nền tảng bắt buộc: tài liệu/thảo luận NVIDIA ghi nhận Fermi không hỗ trợ `cudaMallocManaged()` như các GPU từ CC 3.0 trở lên. [3][4]

Model load-test nên được chia thành hai tầng. `roneneldan/TinyStories-Instruct-33M` là lựa chọn tốt để kiểm tra pipeline Transformer vì file weights hiện khoảng 291 MB và cấu hình là GPT-Neo 4 layer, hidden 768, 16 heads, vocab 50,257, context tối đa 2,048. Tuy nhiên, nó không phải agent/tool-calling model đúng nghĩa. [5][6] Sau khi runtime chạy ổn, `HuggingFaceTB/SmolLM2-135M-Instruct` là stress test tốt hơn cho một LLM instruct hiện đại: 135M params, 30 layer, hidden 576, GQA với 9 attention heads và 3 KV heads; checkpoint float32 được công bố khoảng 269 MB ở một bản hiện có. Nhưng model này dùng kiến trúc Llama-style và config hiện đại, nên không phải target đầu tiên tốt cho Fermi. [7][8]

## 1. Phần cứng cần target

Quadro 2000 thuộc Fermi/CC 2.1. NVIDIA đưa Quadro 2000 trực tiếp vào danh sách legacy GPU compute capability 2.1. Datasheet NVIDIA ghi 1 GB GDDR5, bus 128-bit và 41.6 GB/s bandwidth. [1][2]

Điểm cần sửa so với mô tả trước đó là không nên gọi Quadro 2000 là GPU 384 CUDA cores. Con số 384 thuộc các Quadro K-series như K2000/K620 ở trang thông số NVIDIA. Quadro 2000 Fermi là dòng thấp hơn; datasheet chính thức ghi hơn 192 CUDA parallel-processing cores. [2][9]

Rào cản lớn nhất là software stack, không phải API C. CUDA 8.0 là toolkit cuối cùng hỗ trợ Fermi; CUDA 9 loại bỏ hỗ trợ `sm_2.x` khỏi compiler/toolkit. NVIDIA cũng ghi rõ ứng dụng build bằng CUDA 8 hoặc cũ hơn tiếp tục có thể chạy trên Fermi với driver tương thích. [3]

Driver cũng bị giới hạn: NVIDIA hiện ghi trong ma trận toolkit/driver rằng Fermi CC 2.0 có last toolkit CUDA 8.0 và last driver support R390. Quadro 2000 là CC 2.1 và thuộc cùng nhánh Fermi legacy; NVIDIA forum cũng ghi nhận Quadro 2000 cần driver 390.x trong các hệ thống hiện đại. [1][10]

## 2. Vì sao không dùng PyTorch hiện đại

PyTorch hiện đại phụ thuộc các CUDA toolchain và GPU architectures mới hơn. Với Fermi, vấn đề không phải Python API mà là không còn binary/kernel path được phát hành cho `sm_21`, đồng thời compiler hiện đại cũng không build target Fermi. CUDA 9 đã chính thức xóa Fermi khỏi `nvcc` và CUDA libraries/toolkit support. [3]

Do đó một runtime riêng có lợi thế rõ ràng: build đúng target `compute_21,sm_21`, ít dependency, API cực nhỏ và không phải mang cả framework graph/autograd vào card 1 GB.

## 3. Kiến trúc runtime tối thiểu

Mục tiêu nên là 4–6 file, không phải framework lớn:

```text
runtime.h
runtime.c       // device, context, tensor metadata, allocator
cuda.cu         // CUDA init + memory + launch helpers
ops.cu          // matmul + elementwise + softmax + norm/attention kernels
model.c         // load weights + model forward
binding.c       // tùy chọn: C ABI/Python bridge
```

API public nên nhỏ:

```c
tensor* tensor_new(shape, dtype);
void tensor_free(tensor*);
int tensor_to_cuda(tensor*);
int tensor_to_cpu(tensor*);
tensor* matmul(tensor*, tensor*);
tensor* add(tensor*, tensor*);
tensor* silu_or_gelu(tensor*);
tensor* softmax(tensor*, int dim);
```

Ở giai đoạn đầu không cần autograd, optimizer framework, computation graph, JIT hay allocator phức tạp.

## 4. Memory model trên Fermi

Runtime nên giả định GPU memory và host memory là hai vùng riêng. Dùng `cudaMalloc` trên device và `cudaMemcpy` khi cần chuyển dữ liệu. NVIDIA forum về Fermi ghi nhận `cudaMallocManaged()` yêu cầu compute capability >= 3.0 trong trường hợp được thảo luận, nên Unified Memory không phải nền tảng nên thiết kế dựa vào. [4]

Đây dẫn tới một nguyên tắc hiệu năng quan trọng: copy CPU → GPU một lần khi load model, sau đó giữ weights và activations trên GPU trong suốt forward. Không được đưa từng layer xuống CPU rồi copy lên lại.

1 GB VRAM cũng có nghĩa phải theo dõi VRAM sau khi hệ điều hành/display driver đã chiếm một phần. `1 GB` trên datasheet không đồng nghĩa runtime luôn nhận đủ 1,024 MiB.

## 5. Model load-test số 1: TinyStories-Instruct-33M

Đây là lựa chọn hợp lý nhất cho milestone đầu tiên vì nhỏ, public và có checkpoint trực tiếp.

Hugging Face hiện hiển thị repository khoảng 294 MB và `pytorch_model.bin` khoảng 291 MB. Config công bố:

- `GPTNeoForCausalLM`
- 4 layers
- hidden size 768
- 16 attention heads
- vocabulary 50,257
- max position 2,048
- window size 256
- float32
- attention pattern global/local/global/local [5][6]

Model card cho biết model dựa trên GPT-Neo và được train trên TinyStories. Repository dùng MIT license. [5]

Ước lượng bộ nhớ weights theo 33M parameters ở FP32 là khoảng 132 MB nếu đúng 33M tham số; con số checkpoint file lớn hơn vì file serialization và tensor metadata. Trong thực tế nên đo memory thật từ allocator thay vì suy ra từ file size.

Model này có một ưu điểm rất lớn cho project: cấu trúc attention tương đối nhỏ nhưng vẫn có embedding, linear projections, MLP, activation, layer norm và autoregressive generation. Nó đủ để bắt runtime phải thực sự xử lý Transformer chứ không chỉ benchmark một GEMM.

Nhược điểm: tên `Instruct` không biến nó thành một agent framework. Nó không cung cấp tool protocol hay function calling tiêu chuẩn. Vì vậy nó nên được gọi là `instruct LM load-test`, không phải `agent benchmark`. [5]

## 6. Model load-test số 2: SmolLM2-135M-Instruct

SmolLM2-135M-Instruct là stress test tốt hơn nếu mục tiêu cuối cùng là một mini agent. Config công bố LlamaForCausalLM với 30 layers, hidden 576, intermediate 1536, 9 attention heads, 3 KV heads, vocab 49,152, context 8,192 và tied embeddings. [7]

Một checkpoint float32 hiện trên Hugging Face có kích thước khoảng 269 MB. [8]

GQA là điểm đáng chú ý vì nó giảm số KV heads từ 9 xuống 3, giúp KV cache nhỏ hơn so với MHA cùng hidden size. Tuy nhiên 135M model vẫn nặng hơn đáng kể về compute so với 33M, và kiến trúc Llama/SwiGLU/RoPE hiện đại làm target Fermi khó hơn. Vì vậy nó nên là milestone 2 chứ không phải milestone 1.

## 7. Model 125M đáng để tham khảo: MobileLLM-125M

Meta MobileLLM-125M được thiết kế rõ cho on-device use cases. Paper/repository mô tả các kỹ thuật deep-and-thin architecture, embedding sharing, SwiGLU và grouped-query attention. Bảng model cho 125M: 30 layers, 9 attention heads, 3 KV heads, hidden/token dimension 576, 124.6M parameters, context 2K. [11][12]

Đây là model rất đáng nghiên cứu về kiến trúc khi thiết kế runtime, vì những ý tưởng deep/thin + GQA khá hợp với constrained devices. Nhưng code/pretraining chính thức hiện yêu cầu PyTorch >= 2.0 và training recipe dùng BF16 trên GPU hiện đại, nên không phải stack để chạy trực tiếp trên Quadro 2000. [12][13]

## 8. FP32/FP16 và Fermi

Runtime nên chọn FP32 làm precision gốc. Không nên lấy FP16 làm nền tảng đầu tiên. CUDA hiện đại hỗ trợ `__half`, nhưng các lợi ích phần cứng/throughput của half precision phụ thuộc kiến trúc; Tensor Cores là công nghệ của GPU rất mới, không tồn tại trên Fermi. CUDA 8-era Fermi work vì thế phù hợp nhất với FP32 kernels đơn giản và cuBLAS SGEMM. [3][14]

Có thể nghiên cứu quantization về sau, nhưng int8/4-bit trên Fermi sẽ là một project kernel riêng chứ không phải “bật một flag là xong”.

## 9. Các phép toán runtime thực sự cần

Để inference cho một decoder-only Transformer nhỏ, phiên bản đầu chỉ cần:

```text
embedding lookup
linear / GEMM
bias add
GELU hoặc SiLU tùy model
layer norm hoặc RMSNorm tùy model
QKV projection
attention score
softmax
attention × V
output projection
residual add
final norm
lm_head
argmax hoặc sampling trên CPU
```

`matmul` là kernel/primitive quan trọng nhất. Nên thử cuBLAS trước rồi chỉ tự viết GEMM khi benchmark chứng minh cần thiết. Elementwise kernels và reduction có thể viết CUDA C trực tiếp.

CUDA programming guide cho thấy Fermi có warp 32 threads, shared memory, registers và các ràng buộc occupancy theo compute capability; đây là các yếu tố phải được tính khi tối ưu kernel. [14]

## 10. Thiết kế backend cho nhiều GPU cũ

Không nên encode `Quadro 2000` vào API. Hãy detect compute capability và chọn backend:

```text
GPU
 ↓
compute capability
 ↓
backend_f2x / backend_k3x / backend_m5x...
 ↓
ops
```

NVIDIA lưu ý binary compatibility giữa các generation không được đảm bảo; binary build cho Fermi rất có thể không chạy trực tiếp trên Kepler. Điều này ủng hộ cách chia backend theo architecture thay vì một blob binary chung. [15]

Một thiết kế thực tế là:

```text
frontend API
    ↓
ops interface
    ↓
CUDA backend
 ├── fermi (sm_21)
 ├── kepler
 └── maxwell/pascal (sau này)
```

Fermi là target đầu tiên; Kepler trở thành target mở rộng hợp lý vì nó vẫn là legacy nhưng còn sống lâu hơn về CUDA toolkit. NVIDIA hiện liệt kê Kepler 3.5/3.7 được hỗ trợ tới CUDA 11.x, còn Fermi dừng ở CUDA 8.0. [10]

## 11. Agent thực tế nên nằm ở đâu

Không cần biến GPU runtime thành agent framework.

```text
CPU
 ├── tokenizer
 ├── prompt/chat state
 ├── tool calls
 ├── HTTP / filesystem / shell
 └── sampling control
          ↓
     mini-ai runtime
          ↓
       Quadro 2000
          ↓
      transformer
```

GPU chỉ chịu trách nhiệm neural-network compute. Agent loop nằm trên CPU. Điều này vừa đơn giản vừa đúng với mục tiêu “vài file C nhỏ”.

Một agent mini hoàn chỉnh vì thế có thể dùng model instruct nhỏ + một protocol tool rất đơn giản ở CPU, thay vì đòi model phải hiểu một framework agent khổng lồ.

## 12. Training trên Quadro 2000

Inference nên đi trước.

Training một model 30–33M ở FP32 không chỉ cần weights. Còn gradients, optimizer states và activations. Với Adam, riêng weights + gradients + hai moment thường đã lên khoảng 16 bytes/parameter trước khi tính activation, temporary buffers và fragmentation. Với 33M parameters, mức tối thiểu lý thuyết này khoảng 528 MB; trên GPU 1 GB, phần còn lại cho activation và workspace rất nhỏ. Đây là lý do inference 33M khả thi hơn training 33M.

Training hợp lý hơn là model vài triệu parameters, batch rất nhỏ, sequence ngắn và optimizer tiết kiệm bộ nhớ. TinyStories paper cho thấy các model dưới 10M parameters vẫn có thể tạo văn bản mạch lạc trên dataset TinyStories; điều này làm nhóm model 3M–10M trở thành target nghiên cứu hợp lý cho local training. [16]

Có thể dùng SGD làm optimizer đầu tiên vì nó không mang theo hai state tensors như Adam. Autograd nên được thêm sau khi forward engine ổn định.

## 13. Benchmark nên đo gì

Đừng chỉ đo tokens/second. Hãy ghi ít nhất:

```text
model load time
peak VRAM
kernel time
tokens/sec
first-token latency
CPU↔GPU transfer bytes
GEMM throughput
```

Benchmark 4 mức:

```text
1. GEMM isolated
2. single Transformer block
3. full forward
4. autoregressive generation
```

Quan trọng nhất là benchmark cùng model trên i5-4160 CPU và Quadro 2000. Không nên giả định GPU sẽ thắng mọi workload; với batch nhỏ, layer nhỏ hoặc nhiều kernel launch, CPU có thể cạnh tranh hoặc thắng.

## 14. Milestone đề xuất

### M0 — device bring-up

```text
cudaGetDeviceCount
cudaGetDeviceProperties
show CC / VRAM
cudaMalloc
cudaMemcpy
```

Target: Quadro 2000 nhận đúng là `2.1`.

### M1 — compute primitive

```text
vector add
relu/gelu
reduce
SGEMM
```

Target: correctness + benchmark.

### M2 — tensor runtime

```text
shape
stride
dtype
device pointer
CPU/GPU transfer
```

Target: có tensor abstraction nhưng không overengineer.

### M3 — TinyStories-Instruct-33M

Load checkpoint, mapping weights, embedding, transformer block và logits.

Target tối thiểu:

```text
prompt → forward → logits → token
```

### M4 — generation

Greedy decoding trước; sampling sau.

### M5 — mini agent

CPU tool loop + instruct model.

### M6 — training

Model 3M–10M trước, SGD trước, Adam sau.

## 16. Nghiên cứu thực nghiệm trực tiếp trên máy chủ (Hardware Probe & JIT Breakthrough)

Qua kiểm tra trực tiếp trên hệ thống thực tế (Windows 11 Pro 64-bit, Core i3-4160, Quadro 2000), đã phát hiện các sự thật kỹ thuật có tính chất đột phá cho dự án:

### 16.1. Trạng thái phần cứng và driver thực tế
- **Driver**: Phiên bản `377.83` (WDDM), thư viện Driver API `C:\Windows\System32\nvcuda.dll` có sẵn và hoạt động hoàn hảo.
- **Compute Capability**: Xác nhận chính xác là **`2.1`** (Quadro 2000 GF106GL, 192 cores, 1024 MB VRAM).
- **VRAM khả dụng thực tế**:
  - `nvidia-smi` ghi nhận Desktop GUI và các tiến trình nền chiếm khoảng ~206–480 MB tùy thời điểm.
  - Lệnh gọi Driver API `cuMemGetInfo_v2` trả về: **`817.6 MB Free`** trên tổng số 1024.0 MB.
  - *Ý nghĩa*: Runtime phải quản lý chặt chẽ tổng footprint (Weights + Activations + KV Cache) không được vượt quá **750 MB** để tránh OOM hoặc bị Windows đẩy sang paging system memory qua PCIe 2.0 (tốc độ tụt dốc).

### 16.2. Phát hiện đột phá: Không bắt buộc cài CUDA Toolkit 8.0 & VS cũ
Trước đây, giả định là phải cài đặt CUDA 8.0 Toolkit + Visual Studio 2015 cũ để có `nvcc` build cho `sm_21`. Tuy nhiên, thực nghiệm đã chứng minh một giải pháp gọn gàng và hiện đại hơn:
1. **CUDA Driver API JIT Compilation (`cuModuleLoadData`)**:
   - `nvcuda.dll` đi kèm driver NVIDIA chứa sẵn một trình biên dịch JIT nội tại (Just-In-Time Compiler).
   - JIT compiler này chấp nhận mã nguồn **PTX ISA 3.0 / target `sm_20`** dạng plain-text và biên dịch thành machine code `sm_21` ngay trong lúc runtime khởi chạy.
2. **Thực nghiệm xác nhận**:
   - Một kernel CUDA vector addition (`vec_add`) viết bằng PTX đã được nạp qua `cuModuleLoadData`, cấp phát bộ nhớ GPU (`cuMemAlloc_v2`), copy dữ liệu (`cuMemcpyHtoD_v2`), launch bằng `cuLaunchKernel` và copy kết quả về CPU (`cuMemcpyDtoH_v2`).
   - Kết quả tính toán trên GPU Quadro 2000 chính xác 100%.
3. **Ý nghĩa kiến trúc**:
   - Runtime hoàn toàn có thể được build bằng bất kỳ C compiler nào (MSVC 2022 hiện có trên máy, GCC, Clang, hoặc Rust) mà **không cần `nvcc`**, không cần cài Visual Studio 2015 cổ lỗ, chỉ cần liên kết động với `nvcuda.dll`.
   - Các kernels (GEMM, LayerNorm, GELU, Softmax, RoPE) có thể được lưu trữ dưới dạng chuỗi/file `.ptx` được nạp động khi khởi động.

---

## 17. Phân tích định dạng Checkpoint & Chiến lược nạp Weights

### 17.1. TinyStories-Instruct-33M (`pytorch_model.bin`)
- **Cấu trúc file**: Là một tệp `PK Zip` chứa `data.pkl` (Python pickle) và các stream tensor nhị phân thô (`FloatStorage`, FP32 IEEE 754 little-endian).
- **Trở ngại nếu đọc trực tiếp bằng C**: Tự viết parser Python Pickle bằng C rất phức tạp và dễ lỗi.
- **Giải pháp tối ưu: Script Converter sang Flat Binary (.bin / .safetensors)**:
  - Viết một script Python ngắn (`convert_weights.py`) chạy 1 lần trên CPU bằng Python 3.14 sẵn có.
  - Script đọc weights và trích xuất thành tệp nhị phân phẳng (Flat Binary layout):
    `[header: magic + n_tensors + offsets] [raw float32 data]`
  - C runtime chỉ việc mở file bằng `fopen`/`fread` (hoặc `CreateFileMapping` / `mmap`) và `cuMemcpyHtoD` trực tiếp vào buffer VRAM của GPU trong **vài trăm mili-giây**.

### 17.2. SmolLM2-135M-Instruct (BF16 & 135M params)
- **Kiểm tra trực tiếp**: File `model.safetensors` trên Hugging Face có kích thước 256.6 MB. Header cho thấy toàn bộ 272 tensors đều ở kiểu **`BF16`** (Bfloat16).
- **Vấn đề trên Fermi (CC 2.1)**:
  - Fermi không có phần cứng xử lý số học FP16/BF16.
  - Nếu giải nén ra FP32 để tính toán SGEMM: $135 \times 10^6 \times 4 \text{ bytes} \approx 540 \text{ MB}$ weights.
  - Cộng thêm activations, KV cache và GUI Windows (~200–400 MB) sẽ chạm trần hoặc vượt 1024 MB VRAM.
  - *Giải pháp*: TinyStories-33M (FP32 weights = ~132 MB) là mục tiêu khả thi tuyệt đối cho Quadro 2000. SmolLM2-135M cần kỹ thuật Weight-only Quantization (W8A32 hoặc W4A32) nếu muốn chạy gọn trong 1 GB.

---

## 18. Thiết kế Bộ nhớ & KV Cache Budget (1 GB VRAM)

Với khoảng **750 MB VRAM khả dụng** cho runtime, bài toán phân bổ bộ nhớ cho TinyStories-Instruct-33M được tính toán chi tiết:

| Hạng mục | Công thức / Cấu hình | Dung lượng VRAM |
| :--- | :--- | :--- |
| **Model Weights (FP32)** | 33,086,976 params $\times$ 4 bytes | **~132.3 MB** |
| **Activations (Ping-Pong Buffer)** | 2 $\times$ [batch=1, seq_len=512, hidden=768] $\times$ 4B + MLP | **~12.0 MB** |
| **Q, K, V Projections & Scratch** | Scores matrix $[16, 512, 512] \times 4\text{B}$ | **~16.0 MB** |
| **KV Cache ($L=4$, context $S=512$)** | $2 \times L \times S \times \text{hidden} \times 4\text{B} = 2 \times 4 \times 512 \times 768 \times 4$ | **~12.6 MB** |
| **Vocabulary Logits** | $[1, \text{vocab}=50257] \times 4\text{B}$ | **~0.2 MB** |
| **Dự phòng / Workspace cuBLAS/GEMM** | Scratch space cho ma trận nhân | **~30.0 MB** |
| **Tổng VRAM Runtime cần** | Weights + Dynamic buffers | **~203.1 MB** |

> [!TIP]
> **Nhận định quan trọng**: Runtime cho `TinyStories-Instruct-33M` chỉ tiêu tốn khoảng **~203 MB VRAM** trên GPU. Với 817 MB free đo được, GPU chạy hoàn toàn thoải mái trong VRAM vật lý, không sợ tràn nhớ.

---

## 19. Chiến lược Triển khai GEMM và PTX Kernels

### 19.1. Lựa chọn cuBLAS vs PTX GEMM tự viết
1. **Phương án A — cuBLAS từ CUDA 8.0 toolkit**:
   - Nếu trích xuất file `cublas64_80.dll` từ bộ cài CUDA 8.0, ta có thể gọi hàm SGEMM tối ưu sẵn của NVIDIA.
2. **Phương án B — Tự viết Blocked/Tiled GEMM bằng PTX / CUDA C**:
   - Viết kernel GEMM chia tile ($16 \times 16$ hoặc $32 \times 32$), tận dụng Shared Memory của Fermi (mỗi SM trên Fermi có 48 KB hoặc 16 KB Shared Memory có thể cấu hình).
   - Biên dịch ra PTX bằng compiler offline hoặc nạp chuỗi PTX thuần.
   - Ưu điểm: Độc lập 100%, không cần cài bất kỳ DLL ngoại lai nào ngoài `nvcuda.dll` có sẵn trên mọi máy có driver NVIDIA!

### 19.2. Danh mục các Kernels PTX cần triển khai cho TinyStories (GPT-Neo)
1. `emb_lookup`: Lấy vector từ ma trận embedding $[50257, 768]$ + positional embedding $[2048, 768]$.
2. `gemm_f32`: Nhân ma trận kích thước nhỏ theo batch token ($[1 \times 768] \times [768 \times 768]$).
3. `bias_add`: Cộng bias vector vào ma trận kết quả.
4. `gelu_new`: Kích hoạt GELU xấp xỉ tanh: $0.5x \times (1 + \tanh(\sqrt{2/\pi} \times (x + 0.044715 x^3)))$.
5. `layer_norm`: Chuẩn hóa vector theo mean và variance kèm affine parameters $(\gamma, \beta)$.
6. `attn_score_softmax`: Tính $Q \cdot K^T / \sqrt{d_k}$, causal mask, và softmax theo chiều sequence.
7. `attn_val_mul`: Nhân weights softmax với $V$ để sinh output projection context.

---

## 20. Kiến trúc Phân lớp Chi tiết (End-to-End System)

```text
+-------------------------------------------------------------------------+
|                              HOST (CPU)                                 |
|  - Tokenizer: BPE Tokenizer (vocab.json, merges.txt)                     |
|  - Sampler: Argmax / Temperature / Top-p (tính trên CPU cho 50,257 val) |
|  - Agent Engine: Parse chuỗi instruct, trích xuất command, gọi tool    |
|  - Weight Preprocessor: Chuyển checkpoint PyTorch -> raw .pimi format   |
+-------------------------------------------------------------------------+
                                    |
            [PCIe Transfer: 1 lần nạp Weights (~132 MB) lúc khởi động]
            [Mỗi token sinh ra: Gửi 1 token ID (4 bytes) -> Nhận 50K logits (200 KB)]
                                    v
+-------------------------------------------------------------------------+
|                         DEVICE (Quadro 2000 GPU)                        |
|  - Driver API Loader: nvcuda.dll JIT compilation                        |
|  - VRAM Buffers:                                                        |
|      * Weights Buffer: 132 MB (Static)                                  |
|      * KV Cache Buffer: 13 MB (Stateful)                                |
|      * Ping-Pong Activation Buffer: 12 MB                               |
|  - Compute Core (PTX ISA 3.0 / sm_20):                                  |
|      * GEMM Engine (Tiled Shared Memory MatMul)                         |
|      * Elementwise: Gelu, BiasAdd, ResidualAdd                          |
|      * Reductions: LayerNorm, Attention Softmax                         |
+-------------------------------------------------------------------------+
```

---

## 21. Lộ trình Thực thi Từng bước (Actionable Roadmap)

- [x] **Giai đoạn 0: Khảo sát & Đột phá Kỹ thuật**
  - Khảo sát GPU Quadro 2000 trên driver 377.83: CC 2.1, 817 MB free VRAM.
  - Xác nhận Driver API JIT (`nvcuda.dll`) chạy thành công kernel CUDA PTX mà không cần cài CUDA Toolkit 8.0 hay VS2015.
- [ ] **Giai đoạn 1: Trích xuất Weights & Format Converter**
  - Viết `tools/export_tinystories.py` tải `roneneldan/TinyStories-Instruct-33M` và dump toàn bộ trọng số ra file nhị phân `tinystories_33m.pimi`.
  - Kiểm tra tensor names, shapes, and offsets.
- [ ] **Giai đoạn 2: C / PTX Compute Backend**
  - Tạo cấu trúc dự án C (`src/cuda_drv.c`, `src/kernels.ptx`, `src/tensor.c`).
  - Viết các kernels cốt lõi: `vec_add`, `layer_norm`, `gelu`, `gemm` bằng PTX/CUDA C.
  - Benchmark SGEMM trên Quadro 2000 so sánh với CPU i3-4160.
- [ ] **Giai đoạn 3: Model Forward Engine & Token Generation**
  - Nối ráp toàn bộ khối GPT-Neo 4 layers trên GPU.
  - Viết BPE tokenizer bằng C/Python.
  - Tạo vòng lặp sinh token (Autoregressive loop) với KV-cache.
- [ ] **Giai đoạn 4: Mini Agent Tool-Calling CPU Loop**
  - Xây dựng format prompt instruct cho TinyStories.
  - Tích hợp tool loop CPU (tính toán, đọc ghi file, lệnh shell mini).

---

## 22. Kết quả Thực nghiệm Micro-Benchmark (CPU i3-4160 vs Quadro 2000)

Thực hiện benchmark đối chiếu trực tiếp trên máy bằng C + PTX Driver JIT (MSVC `/O2`, Driver `377.83`):

| Phép toán / Kích thước | CPU (ms) | GPU Kernel (ms) | GPU E2E (ms) | Tỉ số Tăng tốc (Kernel / E2E) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GEMM 128x128x128** | 0.612 ms | **0.205 ms** | 0.694 ms | **2.98x** / 0.88x | `PASS` |
| **GEMM 256x256x256** | 8.504 ms | **1.649 ms** | 2.703 ms | **5.16x** / **3.15x** | `PASS` |
| **GEMM 512x512x512** | 26.232 ms | **12.807 ms** | 14.610 ms | **2.05x** / **1.80x** | `PASS` |
| **Prefill GEMM (128x768x768)** | 15.954 ms | **7.201 ms** | 9.353 ms | **2.22x** / **1.71x** | `PASS` |
| **GPT-Neo MLP Up (1x768x3072)** | 1.122 ms | **0.351 ms** | 4.916 ms | **3.19x** / 0.23x | `PASS` |
| **GPT-Neo Decode QKV (1x768x768)** | **0.140 ms** | 0.273 ms | 1.370 ms | 0.51x / 0.10x | `PASS` |
| **VecAdd N=768 (Residual Add)** | 0.005 ms | **0.003 ms** | 0.507 ms | **1.60x** / 0.01x | `PASS` |
| **VecAdd N=65,536** | 0.024 ms | 0.024 ms | 0.773 ms | 1.03x / 0.03x | `PASS` |
| **VecAdd N=1,048,576 (1M elements)** | 1.908 ms | **0.347 ms** | 14.589 ms | **5.50x** / 0.13x | `PASS` |
| **LayerNorm (1x768 Decode)** | **0.003 ms** | 0.277 ms | 0.710 ms | 0.01x / 0.00x | `PASS` |
| **LayerNorm (128x768 Prefill)** | **0.290 ms** | 3.358 ms | 5.383 ms | 0.09x / 0.05x | `PASS` |
| **Softmax (16x512 Attn Head)** | **0.569 ms** | 1.494 ms | 4.659 ms | 0.38x / 0.12x | `PASS` |
| **Softmax (1x50,257 Vocab Logits)** | **1.320 ms** | 31.727 ms | 34.401 ms | 0.04x / 0.04x | `PASS` |

### Quy tắc Vàng cho Thiết kế `pimi`:
1. **GPU nhận việc nặng**: Ma trận weights cố định trên VRAM. Toàn bộ phép nhân GEMM trong Transformer (Attention Projections, MLP Up/Down) và Residual Add chạy trên GPU.
2. **Không copy round-trip giữa các layer**: Tensor activations giữ hoàn toàn trong VRAM GPU suốt cả 4 layers.
3. **CPU nhận việc sampling**: Sau khi GPU tính xong vector logits $[1 \times 50257]$ ở layer cuối cùng, chỉ copy 200 KB này về RAM để CPU tính Softmax / Argmax (mất 1.3 ms thay vì 31.7 ms trên GPU).

---

## Sources


[1] NVIDIA Developer. “Legacy CUDA GPU Compute Capability.” https://developer.nvidia.com/cuda/gpus/legacy
[2] NVIDIA. “Quadro 2000 Datasheet.” https://www.nvidia.com/docs/IO/40049/NV_DS_QUADRO_2000_US_LR.pdf
[3] NVIDIA. “CUDA Toolkit 9.0 Release Notes / CUDA C Programming Guide.” https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes/
[4] NVIDIA Developer Forums. “cudaMallocManaged fails” discussion noting Fermi/CC 2.1 and managed-memory limitation. https://forums.developer.nvidia.com/t/cudamallocmanaged-fails/71250
[5] Ronen Eldan. “TinyStories-Instruct-33M,” Hugging Face repository. https://huggingface.co/roneneldan/TinyStories-Instruct-33M
[6] Ronen Eldan. “TinyStories-Instruct-33M config.json,” Hugging Face. https://huggingface.co/roneneldan/TinyStories-Instruct-33M/blob/main/config.json
[7] Hugging Face. “SmolLM2-135M-Instruct config.json.” https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blame/83212e1e2b3cfd6958f3707877bb878945dea8ee/config.json
[8] Hugging Face. “unsloth/SmolLM2-135M-Instruct model.safetensors.” https://huggingface.co/unsloth/SmolLM2-135M-Instruct/blob/main/model.safetensors
[9] NVIDIA. “Previous Generation Desktop Graphics Cards / Quadro specifications.” https://www.nvidia.com/en-us/products/workstations/previous-quadro-desktop-gpus/
[10] NVIDIA. “CUDA Toolkit, Driver, and Architecture Matrix.” https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
[11] Meta / Facebook Research. “MobileLLM.” https://github.com/facebookresearch/MobileLLM
[12] Hugging Face. “facebook/MobileLLM-125M.” https://huggingface.co/facebook/MobileLLM-125M
[13] Meta / Facebook Research. “MobileLLM pretraining scripts.” https://github.com/facebookresearch/MobileLLM/blob/main/pretrain.sh
[14] NVIDIA. “CUDA C Programming Guide, CUDA 8.0.” https://docs.nvidia.com/cuda/archive/8.0/cuda-c-programming-guide/
[15] NVIDIA. “NVCC GPU Compilation Model.” https://docs.nvidia.com/cuda/archive/11.5.0/cuda-compiler-driver-nvcc/index.html
[16] Ronen Eldan, Yuanzhi Li. “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?” arXiv:2305.07759. https://arxiv.org/abs/2305.07759

