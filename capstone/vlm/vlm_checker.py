import json
import re
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from capstone.vlm.prompts import PICK_PROMPT, HANDOVER_PROMPT


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


class VLMChecker:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[VLM] model_id = {self.model_id}")
        print(f"[VLM] device = {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

    def _extract_json(self, text: str) -> Dict:
        """
        VLM 출력에서 JSON 부분만 뽑는다.
        모델이 앞뒤에 설명을 붙이는 경우를 대비한다.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return {
                "state": "UNKNOWN",
                "confidence": 0.0,
                "reason": "VLM output did not contain JSON.",
                "raw": text,
            }

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "state": "UNKNOWN",
                "confidence": 0.0,
                "reason": "Failed to parse VLM JSON output.",
                "raw": text,
            }

        state = data.get("state", "UNKNOWN")
        confidence = data.get("confidence", 0.0)
        reason = data.get("reason", "")

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return {
            "state": state,
            "confidence": confidence,
            "reason": reason,
            "raw": text,
        }

    def check_image(self, image_path: str, mode: str) -> Dict:
        """
        mode:
        - pick: 약통 집기 가능 여부 판단
        - handover: 사람에게 건네주기 가능 여부 판단
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return {
                "state": "UNKNOWN",
                "confidence": 0.0,
                "reason": f"Image not found: {image_path}",
            }

        image = Image.open(image_path).convert("RGB")

        if mode == "pick":
            prompt = PICK_PROMPT
        elif mode == "handover":
            prompt = HANDOVER_PROMPT
        else:
            raise ValueError("mode must be 'pick' or 'handover'")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )

        output = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return self._extract_json(output)