언어의 헌법 v0.1

1. 기계가 생성·검증·최적화하기 쉽고, 인간이 바이너리까지 역추적할 수 있어야 한다.
Human IR
   ↓
Canonical IR
   ↓
Binary IR
세 표현은 서로 다른 언어가 아니라 동일한 의미를 가진 세 가지 표현이다.
2. 첫 번째 Opcode 체계

00000000  NOP
00000001  INPUT
00000010  OUTPUT
00000011  CONST
00000100  ADD
00000101  SUB
00000110  MUL
00000111  DIV
00001000  MATMUL
00001001  TRANSPOSE
00001010  RESHAPE
00001011  REDUCE
00001100  COPY
00001101  ALLOC
00001110  FREE

8-bit opcode로 시작

3. Tensor도 언어의 기본 객체로 만든다

4. Device도 언어 자체에서 표현: machine-readable identifier

5. Graph ID 어떤 노드도 이름이 아니라 ID와 dependency로 연결된다.

6. 인간이 명세서를 가지고 0101 → 의미를 역추적할 수 있다.

7. 양방향 변환이 가능해야 한다

8. 


Human - AI - IR - Optimizer - Compiler - Binary - Hardware.
