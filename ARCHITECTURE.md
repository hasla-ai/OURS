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

## OUR-MIR v1.0 — 공식 DType 시스템

FP32 -> DTYPE_FP32 = 0x04 : IR 자체가 타입 ID.
INT8, INT32, FP16, BF16, FP32 을 지원.
0101 형태까지 인간이 역추적할 수 있어야 한다는 계약의 유지.

DType.FP32 -> 0x06

INT8 -10 -> f6 -> -10
INT8 127 -> 7f -> 127
INT32 100000 -> a0860100 -> 100000
FP32 1.0 -> 0000803f -> 1.0
FP32 3.14 -> c2f54840 -> 3.1399998664855957

## v1.1 — Tensor Memory Layout
 - Tensor를 메모리 관점에서 완전히 정의.
 Tensor
 ├── dtype
 ├── shape
 ├── strides
 ├── address
 └── nbytes

 Tensor[1, 2] -> base_address + offset.

 TENSOR LAYOUT: Tensor = metadata + memory region.
 - GPU로 가도 그대로 유지
 - CPU, GPU, NPU, TPU 동일한 Tensor/IR 추상화 아래에서 관리

              Tensor
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
     Shape    DType    Address
       │        │        │
       └────────┼────────┘
                ↓
         Memory Manager
                ↓
           Raw Bytes

shape: (2, 3)
dtype: FP32
address: 0x1000
strides: (3, 1)
elements: 6
bytes: 24

-> FP32 = 4 bytes
6 elements × 4 = 24 bytes
 
Tensor              -> address = 0x1000, [0x1000  FP32, 0x1004  FP32, 0x1008  FP32]
shape = [2,3]                            [0x100C  FP32, 0x1010  FP32, 0x1014  FP32]
dtype = FP32

실제 연속 메모리 레이아웃으로 정의.

Tensor[0,0] → 0x1000
Tensor[0,1] → 0x1004
Tensor[0,2] → 0x1008

Tensor[1,0] → 0x100C
Tensor[1,1] → 0x1010
Tensor[1,2] → 0x1014

## OUR Runtime v1.2 — Tensor Read/Write
  인간이 코딩하는 것이 아니라 기계와 기계가 서로 계산 프로그램을 생성하고 교환한다는 방향의 기반

tensor[1, 2] = 7.5
기존 TensorLayout에 메모리 연결과 원소 단위 read/write를 추가

tensor.write((1,2), 7.5) -> offset (1 × 3 + 2= 5) -> 5 × 4 bytes= 20
-> 0x1000 + 20= 0x1014 -> FP32 encode(7.5) -> 4 bytes -> Memory[0x1014:0x1018]

읽을 때 역방향으로: Memory -> 4 bytes -> FP32 decode -> 7.5

## OUR-MIR v1.3 — LOAD / STORE Instruction

tensor.read() / tensor.write() 대신 메모리 접근도
 - LOAD : LOAD memory → register/tensor
 - STORE : STORE value → memory

단, 현재 Instruction 포맷을 그대로 활용.
64 bit
[ OPCODE ][ A ][ B ][ OUTPUT ][ FLAGS ]
   8       16   16      16        8

LOAD 10 → 20    => input_a = 10, output  = 20

Human / AI -> OUR-MIR -> LOAD / STORE -> Runtime -> Memory

TensorValue 가 tensor_id, address, size 이므로 
LOAD/STORE가 어떤 원소를 대상으로 하는지 표현할 수 없으므로, 

Instruction
    │
    ├── opcode
    ├── dtype
    ├── operands[]
    └── attributes{}

    64bit 부족하므로 가변 길이 IR로 설계변경. 

                   OUR Language
                        │
                        ↓
                  OUR-MIR Graph
                        │
              ┌─────────┴─────────┐
              ↓                   ↓
         Human-readable       Binary MIR
              │                   │
              │                   ↓
              │              Lowering
              │                   │
              └──────────────→ Machine IR
                                  │
                                  ↓
                              CPU/GPU

                              로 구조 변경.    

## OUR-MIR v1.4 — Variable-Length Instruction / Operand System

## 1) Variable-Length Instruction

Instruction
├── opcode
├── operands[]
├── result[]
└── attributes{}

즉 

STORE
  tensor = 17
  index  = [4, 8, 2]
  dtype  = FP32
  value  = 7.5 를 Binary로 직렬화.

OP STORE
  OPERANDS
    Operand(tensor, 17)
    Operand(index, [4, 8, 2])
    Operand(value, 7.5)
  RESULTS
  ATTRIBUTES
    dtype=FP32

STORE Tensor17[4,8,2] = 7.5 를 IR이 정확하게 표현.

Operand: 계산에 참여하는 값, Tensor, Scalar, Index, Memory
Attribute: 계산 방법을 설명하는 메타데이터, dtype=FP32, transpose=false, axis=1

를 분리했다.

## 2) Binary Encoding : 
Binary는 인간이 읽기 어려워도 되지만, 인간이 완전히 해석 불가능해서는 안 된다.
 - Text IR, Binary IR, Disassembler 항상 같이.
Binary가 의미를 잃지 않도록 한다.

STORE
  tensor=17
  index=[4,8,2]
  value=7.5
  dtype=FP32

  -> [TVL Structure: TYPE, LENGTH, VALUE]: 앞으로 Operand 종류가 늘어나도 확장 가능

STORE 라는 의미가 실제 byte sequence로 내려간다.

Human
  │
  ▼
STORE tensor=17 index=[4,8,2]
  │
  ▼
OUR-MIR
  │
  ▼
TLV Binary
  │
  ▼
010101010...

다른 기계는 Binary를 받아 다시:

010101...
   ↓
TLV
   ↓
IRInstruction
   ↓
STORE
tensor=17
index=[4,8,2]
로 복원할 수 있다.

                OUR AI LANGUAGE
                       │
                       ▼
                 OUR-MIR
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Opcode       Operand      Attribute
          │            │            │
          └────────────┼────────────┘
                       ▼
                  IR Graph
                       │
                       ▼
                 Binary MIR
                       │
                       ▼
                    Runtime
                       │
                       ▼
                    Memory

