import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sounddevice as sd  # 🎯 실시간 오디오 장치 검색을 위해 추가

from .audio_io import AudioInput
from .stt_engine import STTEngine


class SenseVoiceSTTNode(Node):
    def __init__(self):
        super().__init__("sensevoice_stt_node")

        self.declare_parameter(
            "model_dir",
            "/home/lerobot/aicapstone/lerobot/capstone/gengen/models/sensevoice_ko",
        )
        self.declare_parameter("model_name", "model.onnx")
        self.declare_parameter("language", "ko")
        self.declare_parameter("device", "cpu")

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("channels", 2)  # 🎯 ReSpeaker Lite 오디오 규격에 맞춰 기본값 2로 변경
        self.declare_parameter("frame_ms", 30)
        self.declare_parameter("silence_duration_ms", 1000)
        self.declare_parameter("min_speech_ms", 300)
        self.declare_parameter("pre_roll_ms", 200)
        self.declare_parameter("energy_threshold", 0.01)

        # 🎯 기본값을 -1로 두어, 밖에서 번호를 안 주면 자동으로 이름을 찾도록 설계
        self.declare_parameter("audio_device", -1)
        self.declare_parameter("use_channel_index", 0)
        self.declare_parameter("publish_topic", "/stt/text")

        model_dir = self.get_parameter("model_dir").value
        model_name = self.get_parameter("model_name").value
        language = self.get_parameter("language").value
        device = self.get_parameter("device").value

        sample_rate = int(self.get_parameter("sample_rate").value)
        channels = int(self.get_parameter("channels").value)
        frame_ms = int(self.get_parameter("frame_ms").value)
        silence_duration_ms = int(self.get_parameter("silence_duration_ms").value)
        min_speech_ms = int(self.get_parameter("min_speech_ms").value)
        pre_roll_ms = int(self.get_parameter("pre_roll_ms").value)
        energy_threshold = float(self.get_parameter("energy_threshold").value)

        audio_device_param = int(self.get_parameter("audio_device").value)
        use_channel_index = int(self.get_parameter("use_channel_index").value)
        publish_topic = self.get_parameter("publish_topic").value

        # 🎯 [하드코딩 없는 정석 로직] 장치 번호 자동 검색 및 예외 처리
        audio_device = None
        if audio_device_param >= 0:
            # 밖에서 명시적으로 0 이상의 포트 번호를 인자로 주었을 때는 그 번호를 존중
            audio_device = audio_device_param
            self.get_logger().info(f"Using explicitly requested audio_device index: {audio_device}")
        else:
            # 인자가 없거나 -1일 경우, 시스템 전체 장치를 스캔하여 'ReSpeaker'를 자동으로 추적
            self.get_logger().info("Searching for ReSpeaker device automatically...")
            for i, d in enumerate(sd.query_devices()):
                if "ReSpeaker" in d["name"] and d["max_input_channels"] > 0:
                    audio_device = i
                    break
            
            if audio_device is not None:
                self.get_logger().info(f"✨ Successfully auto-detected ReSpeaker at index [{audio_device}]")
            else:
                self.get_logger().warn("⚠️ ReSpeaker not found in device list. Falling back to system default.")
                audio_device = None  # None이면 sounddevice가 시스템 기본 마이크를 잡습니다.

        self.text_pub = self.create_publisher(String, publish_topic, 10)

        self.audio = AudioInput(
            sample_rate=sample_rate,
            channels=channels,
            frame_ms=frame_ms,
            silence_duration_ms=silence_duration_ms,
            min_speech_ms=min_speech_ms,
            pre_roll_ms=pre_roll_ms,
            energy_threshold=energy_threshold,
            device=audio_device,
            use_channel_index=use_channel_index,
        )

        self.engine = STTEngine(
            model_dir=model_dir,
            model_name=model_name,
            language=language,
            device=device,
        )

        self._running = True
        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()

        self.get_logger().info("SenseVoice generic STT node started")
        self.get_logger().info(f"model_dir={model_dir}")
        self.get_logger().info(f"publish_topic={publish_topic}")
        self.get_logger().info(
            f"Final Mapped -> audio_device={audio_device}, channels={channels}, use_channel_index={use_channel_index}"
        )

    def _run_loop(self):
        while rclpy.ok() and self._running:
            try:
                wav_path = self.audio.record_utterance()
                text = self.engine.transcribe(wav_path)

                if not text:
                    self.get_logger().info("(인식 결과 없음)")
                    continue

                msg = String()
                msg.data = text
                self.text_pub.publish(msg)

                self.get_logger().info(f"recognized: {text}")

            except Exception as e:
                self.get_logger().error(f"STT loop error: {e}")
                time.sleep(0.2)

    def destroy_node(self):
        self._running = False
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SenseVoiceSTTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
