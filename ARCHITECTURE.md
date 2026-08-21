1. OUR-MIR v0.1 Binary Encoder/Decoder
  기계가 생성할 수 있고, 바이너리로 저장할 수 있으며, 다시 인간이 의미를 복원할 수 있는 우리만의 명령 표현.(`ourmir.py`)
  MATMUL Tensor1 Tensor2 → Tensor3를 우리 바이너리로 바꾸고 다시 복원한다.
Human IR
   ↓
Encoder
   ↓
64-bit Binary
   ↓
Decoder
   ↓
Human IR

=== ORIGINAL ===
MATMUL 1 2 -> 3 [flags=0]

=== BINARY ===
00001000...

=== DECODED ===
MATMUL 1 2 -> 3 [flags=0]

=== CHECK ===
True