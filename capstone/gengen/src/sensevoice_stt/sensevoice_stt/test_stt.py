import sys
from .stt_engine import STTEngine


def main():
    if len(sys.argv) < 2:
        print("usage:", sys.argv[0], "/real/path/to/file.wav")
        return

    wav_path = sys.argv[1]
    print("wav_path =", wav_path)

    engine = STTEngine(
        model_dir="/home/lerobot/aicapstone/lerobot/capstone/gengen/models/sensevoice_ko",
        model_name="model.onnx",
        language="ko",
        device="cpu",
    )

    text = engine.transcribe(wav_path)

    print("result_repr =", repr(text))
    print("result =")
    print(text)


if __name__ == "__main__":
    main()
