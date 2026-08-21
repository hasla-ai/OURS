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

2. OUR-MIR v0.2 — 계산 그래프 (`graph.py`)
기존 Instruction을 건드리지 않고, 그 위에 Graph / Node / Tensor 계층을 추가
Tensor 1 ─┐
          ├─ MATMUL ─→ Tensor 3
Tensor 2 ─┘
를 기계관리가 가능하다.

3. OUR-MIR v0.3 — Shape Type System으로 간다.

이번에는 기존 코드를 최대한 건드리지 않고 Graph.validate()에 MATMUL의 shape 검증과 출력 shape 추론을 추가.

  AI가 IR 생성
       ↓
   Shape Check
       ↓
   Type Check
       ↓
 Graph Validation
       ↓
     실행

4. OUR-MIR v0.4 — DType Type System
  - 이번 단계부터는 단순히 shape만 맞는 게 아니라 데이터 타입까지 IR이 검증.

  - 안전한 기계간 프로토콜로 발전하기 시작. F32 × F16 같은 문제까지 잡기 위해 dtype 시스템을 넣는다.
  - 그 다음에는 ADD, MUL, RESHAPE, TRANSPOSE의 shape/type inference를 추가해서 Tensor 연산 전체를 IR 차원에서 검증함.

  - shape만 맞는 게 아니라 데이터 타입까지 IR이 검증함.

Opcode → Tensor ID → Shape → DType 까지 기계적으로 검증.

5. OUR-MIR v0.5 — Type Inference: Tensor 3의 타입과 shape 추론

  지금은 AI가 반드시 Tensor 3 = FP32 [2,4]를 직접 선언해야.
  다음부터는 AI가: MATMUL 1 2 → 3만 생성하면 Validator가 자동으로 Tensor 3, shape = [2,4], dtype = FP32
를 추론하도록.

  

