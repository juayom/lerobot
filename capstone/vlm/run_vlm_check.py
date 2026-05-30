# capstone/vlm/run_vlm_check.py

import argparse
import json

from capstone.vlm.vlm_checker import VLMChecker


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="VLM으로 판단할 RGB 이미지 경로",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["pick", "handover"],
        help="pick 또는 handover",
    )

    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Hugging Face VLM model id",
    )

    args = parser.parse_args()

    checker = VLMChecker(model_id=args.model_id)
    result = checker.check_image(args.image, args.mode)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()