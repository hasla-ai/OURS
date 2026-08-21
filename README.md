## AI가 다른 AI 시스템을 구축하고 최적화하기 위한 기계간 프로그래밍·컴파일·실행 체계
 - ## Tensor 연산을 opcode화
 - ## 특정 회사 GPU에 종속되지 않는 IR

AI 컴퓨팅 스택 자체를 0부터 만든다.
우리는 LLM을 만드는 게 아니라, LLM을 만들 수 있는 컴퓨터 언어와 AI 컴퓨팅 스택부터 만든다.
Python은 호스트 언어일 뿐이고, 우리가 만드는 언어 자체에는 Python 문법이나 기존 AI 프레임워크를 사용하지 않는다. 그러면 언어의 핵심 primitive도 달라진다. 이것은 인간이 코딩하지 않고 기계간 코딩을 위한.

기계 ↔ 기계가 AI 시스템을 구성·최적화·실행하기 위한 Machine-to-Machine programming language
더 나아가서는 명령형 코드 자체보다 계산 그래프/IR(Intermediate Representation) 중심으로.
 - 무엇을 계산할지 데이터 흐름(What)을 연결한 지도"를 먼저 만드는 방식. 실행 전 전체 계산 과정을 자동 최적화하기 쉬움.

 Machine-to-Machine / AI 컨트롤러에 IR 중심이 필요한가?

기계나 AI 컨트롤러가 시스템을 제어할 때 IR 형태가 유리한 이유는 다음과 같습니다.

하드웨어 입체 최적화: 실행하기 전에 계산 전체 지도를 훑어보고, "이 덧셈과 곱셈은 묶어서 GPU Tensor Core로 보내고, 데이터 로딩은 CPU DRAM에서 미리 Prefetch하자"와 같이 하드웨어 맞춤형 병렬화/스케줄링을 미리 계산할 수 있습니다.

불필요한 연산 삭제 및 융합 (Kernel Fusion): 그래프 상에서 연속된 연산(예: Attention 안의 QKᵀ 곱셈과 Softmax)을 하나의 효율적인 GPU 커널 명령으로 합쳐버릴 수 있습니다.

기계 생성/검증의 용이성: LLM이나 AI Agent는 텍스트 형태의 복잡한 코드보다, 규칙화된 데이터 구조(노드와 엣지로 이루어진 IR Graph)를 훨씬 쉽게 생성하고 유효성을 검증할 수 있습니다.

그리고 이 Machine-to-Machine / AI 컨트롤러에 대한 IR 중심 논리를

전체 Personal PC 등으로 확대한다. 단순한 '프로그램 실행기'에서 '기계가 스스로 연산을 최적화하여 배치하는 지능형 리소스 풀'로의 전환.

기존 PC의 OS(Windows, macOS 등)는 개발자가 작성한 바이너리를 명령형으로 순차 실행하지만, IR 중심 PC는 모든 작업(OS, 앱, AI 작업)을 하나의 거대한 계산 그래프로 처리.

# 변수 → 함수 → 타입 → 배열 → Tensor → compiler
순으로 개발.

Human
  │
  │ 목표 / 제약 / 요구
  ↓
AI Agent
  │
  │ 우리 언어로 프로그램 생성
  ↓
Machine Language
  │
  ├── Tensor
  ├── Memory
  ├── Kernel
  ├── Parallelism
  ├── GPU
  ├── Gradient
  └── Model Graph
  ↓
AI Computing Runtime
  ↓
CPU / GPU / NPU

## [목표: AI가 자기 코드를 생성 → 실행 → 성능 측정 → 수정 → 재실행하는 폐쇄 루프]

AI
 ↓
Machine Code / IR 생성
 ↓
Compiler
 ↓
Runtime
 ↓
CPU/GPU 실행
 ↓
Performance / Error
 ↓
AI가 분석
 ↓
Code 수정

## Binary-native, human-auditable.

인간이 readable 해야함.

Human-readable IR
        ↓
Canonical IR
        ↓
Binary IR

1) 첫째, 첫 번째로 만들어야 하는 것은 우리 IR specification.
2) 둘째, 우리 언어의 opcode, tensor type, memory model, graph node, binary encoding 규칙을 설계하는 것, 언어 헌법.



모든 프로그램은 Graph다.
Tensor는 이름보다 ID가 우선이다.

DType 

0000 = BOOL
0001 = INT8
0010 = INT16
0011 = INT32
0100 = INT64
0101 = FP16
0110 = BF16
0111 = FP32
1000 = FP64

중요한 점은 IR이 특정 회사의 GPU에 종속되지 않는 것.


## Binary 구조
모든 Instruction은 v0.1에서 고정 길이 구조를 사용한다. 56 bit.
┌─────────┬──────────┬──────────┬──────────┐
│ OPCODE  │ INPUT A  │ INPUT B  │ OUTPUT   │
│  8 bit  │ 16 bit   │ 16 bit   │ 16 bit   │
└─────────┴──────────┴──────────┴──────────┘

다만 실제 저장은 byte alignment를 위해 64 bit instruction word로.

[ OPCODE ][ INPUT_A ][ INPUT_B ][ OUTPUT ][ FLAGS ]
   8         16          16         16        8


Canonical IR: Binary보다 사람이 보기 쉽게 표현할 수 있다.
Canonical IR을 AI가 직접 생성한다.

검증기: IR에는 반드시 Validator가 붙는다. 생성한 코드도 실행 전에 스스로 검증한다.



최종적으로 AI가 이렇게 만들 수 있어야 한다.

AI
 ↓
Canonical IR 생성
 ↓
Validator
 ↓
Graph Optimizer
 ↓
Binary IR
 ↓
Backend Compiler
 ↓
CPU / GPU / NPU

그리고 실행 후:

performance
memory
latency
error

를 다시 AI에게 전달한다.

              ┌──────────────┐
              │      AI      │
              └──────┬───────┘
                     ↓
                  OUR IR
                     ↓
                 VALIDATOR
                     ↓
                 OPTIMIZER
                     ↓
                  BINARY
                     ↓
              CPU / GPU / NPU
                     ↓
             RESULT / METRIC
                     │
                     └──────────→ AI

이게 우리가 처음부터 이야기한 기계간 코딩의 핵심이다.




