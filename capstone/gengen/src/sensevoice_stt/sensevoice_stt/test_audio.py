from .audio_io import AudioInput


def main():
    audio = AudioInput(
        sample_rate=16000,
        channels=6,
        frame_ms=30,
        silence_duration_ms=1000,
        min_speech_ms=300,
        pre_roll_ms=200,
        energy_threshold=0.01,
        device=0,
        use_channel_index=0,
    )

    print("말해봐. 문장이 끝나면 wav 저장함.")
    wav_path = audio.record_utterance()
    print(wav_path)


if __name__ == "__main__":
    main()
