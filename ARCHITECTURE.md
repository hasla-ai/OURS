## OUR-MIR v0.1 Binary Encoder/Decoder
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

## OUR-MIR v0.2 — 계산 그래프 (`graph.py`)
기존 Instruction을 건드리지 않고, 그 위에 Graph / Node / Tensor 계층을 추가
Tensor 1 ─┐
          ├─ MATMUL ─→ Tensor 3
Tensor 2 ─┘
를 기계관리가 가능하다.

## OUR-MIR v0.3 — Shape Type System으로 간다.

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

## OUR-MIR v0.4 — DType Type System
  - 이번 단계부터는 단순히 shape만 맞는 게 아니라 데이터 타입까지 IR이 검증.

  - 안전한 기계간 프로토콜로 발전하기 시작. F32 × F16 같은 문제까지 잡기 위해 dtype 시스템을 넣는다.
  - 그 다음에는 ADD, MUL, RESHAPE, TRANSPOSE의 shape/type inference를 추가해서 Tensor 연산 전체를 IR 차원에서 검증함.

  - shape만 맞는 게 아니라 데이터 타입까지 IR이 검증함.

Opcode → Tensor ID → Shape → DType 까지 기계적으로 검증.

## OUR-MIR v0.5 — Type Inference: Tensor 3의 타입과 shape 추론

  지금은 AI가 반드시 Tensor 3 = FP32 [2,4]를 직접 선언해야.
  다음부터는 AI가: MATMUL 1 2 → 3만 생성하면 Validator가 자동으로 Tensor 3, shape = [2,4], dtype = FP32
를 추론하도록.

  (`test.py`) Tensor 3을 직접 정의하지 않았는데 엔진이 추론함.

AI
 ↓
IR 생성
 ↓
Type Inference
 ↓
Shape Inference
 ↓
Validation
 ↓
실행 기초.

## OUR-MIR v0.6 — Dependency Graph + Topological Execution

계산 순서 : graph.nodes.values() 순서 -> dependency를 분석해서 N1 → N2 → N3 순서로 재배열.
DAG(Directed Acyclic Graph) 개념 구현.


(생성 순서)
N3 = ADD N1 N2
N1 = MATMUL A B
N2 = MUL C D

(dependency 분석)
N1 ──┐
     ├──→ N3
N2 ──┘
실행 순서를: N1 → N2 → N3로 결정.

NODES
  Node(id=2, ADD 3 4 -> 5 [flags=0])
  Node(id=1, MATMUL 1 2 -> 3 [flags=0])

EXECUTION PLAN
0: NODE 1 MATMUL 1 2 -> 3 [flags=0]
1: NODE 2 ADD 3 4 -> 5 [flags=0]

## OUR-MIR v0.7 — 자체 CPU Runtime

MATMUL, ADD, MUL에 대한 Execution Plan을 받아서 CPU Runtime이 직접 실행함.
vertical slice: IR → Graph → Execution Plan → 실제 계산함.

이번 단계부터 IR이 실제로 숫자를 계산.
(`runtime.py`)

## v 0.8 Memory Runtime
Python의 리스트를 Runtime의 핵심 저장소로 사용하지 않는다. CPU/GPU 메모리 모델의 기반
Tensor -> OUR Memory Manager -> Memory Block -> Raw bytes

()`test_memory.py`)
b'\x01\x02\x03\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
MEMORY
  MemoryBlock(address=1, size=16, active=True)

## OUR-MIR v0.9 — 자체 DType Encoding

## <실제 데이터 표현 규칙>
## Tensor -> FP32 -> 32-bit representation -> 4 bytes -> Memory

Runtime이 IEEE-754 비트 구조로 직접 encode/decode : INT32, FP16, BF16, FP32
FP32 -> 32 bits -> Memory bytes. 

IR -> Tensor -> DType -> Raw Memory -> CPU Operation
Tensor → Python float / Python list 의존도 탈출.

(`dtype.py`)

test (`test_dtype.py`)

0.0 -> 00000000 -> 0.0
1.0 -> 0000803f -> 1.0
-1.0 -> 000080bf -> -1.0
2.5 -> 00002040 -> 2.5
10.0 -> 00002041 -> 10.0