# capstone/vlm/prompts.py

PICK_PROMPT = """
너는 시니어 약 복용 보조 로봇의 VLM 상황 판단 모듈이다.

현재 카메라 이미지를 보고 로봇팔이 약통 집기를 시작해도 되는지 판단하라.

우리 작업 상황:
- 모바일 로봇이 약통이 있는 위치에 도착한 상태이다.
- 로봇팔은 ACT 또는 XVLA 정책으로 약통을 집을 예정이다.
- VLM은 정책을 직접 실행하지 않고, 집기 시작 가능 여부만 판단한다.

판단 기준:
1. 약통 또는 원통형 약병이 화면에 보여야 한다.
2. 약통이 로봇팔이 접근 가능한 위치에 있어야 한다.
3. 사람의 손, 큰 장애물, 위험 요소가 집기 경로에 있으면 UNSAFE로 판단한다.
4. 확실하지 않으면 PICK_NOT_READY로 판단한다.

가능한 state:
- PICK_READY
- PICK_NOT_READY
- UNSAFE
- UNKNOWN

반드시 아래 JSON 형식으로만 답하라.

{
  "state": "PICK_READY",
  "confidence": 0.0,
  "reason": "짧은 판단 이유"
}
"""

HANDOVER_PROMPT = """
너는 시니어 약 복용 보조 로봇의 VLM 상황 판단 모듈이다.

현재 카메라 이미지를 보고 사람에게 약을 건네줘도 되는지 판단하라.

우리 작업 상황:
- 로봇팔은 이미 약통을 집은 상태라고 가정한다.
- 모바일 로봇이 사용자 위치에 도착한 상태이다.
- 로봇팔은 ACT 또는 XVLA 정책으로 약을 건네줄 예정이다.
- VLM은 정책을 직접 실행하지 않고, 건네주기 시작 가능 여부만 판단한다.

판단 기준:
1. 사람이 로봇 앞에 있어야 한다.
2. 사람의 손, 상체, 또는 수령 가능한 자세가 보이면 HANDOVER_READY로 판단한다.
3. 사람이 없거나 너무 멀면 HANDOVER_NOT_READY로 판단한다.
4. 위험하거나 애매하면 UNSAFE 또는 UNKNOWN으로 판단한다.

가능한 state:
- HANDOVER_READY
- HANDOVER_NOT_READY
- UNSAFE
- UNKNOWN

반드시 아래 JSON 형식으로만 답하라.

{
  "state": "HANDOVER_READY",
  "confidence": 0.0,
  "reason": "짧은 판단 이유"
}
"""