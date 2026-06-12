import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import glob
import time
import random
from gtts import gTTS
from datetime import datetime

class DialogControllerNode(Node):
    def __init__(self):
        super().__init__("dialog_controller_node")
        self.stt_sub = self.create_subscription(String, "/stt/text", self.stt_callback, 10)
        self.arm_pub = self.create_publisher(String, "/robot_arm/command", 10)
        self.tts_pub = self.create_publisher(String, "/robot_tts/command", 10)

        # 상태 제어 변수
        self.is_awakened = False       # False: 사용자가 부르기 전 대기 상태
        self.is_greeting_playing = False
        self.is_delivering = False
        self.response_received = False
        self.is_medication_time = False # TRUE: CASE 1, FALSE: CASE 2 (시연장 대응 기본값)

        self.timeout_timer = None
        self.listen_ready_time = 0.0  # 로봇 음성 출력 직후 STT 오인식 방지용
        self.last_spoken_text = ""    # 로봇이 마지막으로 말한 문장 저장용

        # 12시 / 18시 자동 복약 알림용
        self.last_auto_alert_key = None
        self.auto_alert_timer = self.create_timer(30.0, self.check_medication_auto_alert)

        self.get_logger().info("🤖 [gengen 말하기] 사용자 호출 대기 모드 노드가 시작되었습니다.")
        self.get_logger().info("👂 [대기] 시연을 시작하려면 마이크에 대고 '젠젠아'라고 불러주세요...")

    def trigger_opening_flow(self):
        """ 사용자가 '젠젠아'라고 부르면 시작되는 오프닝 흐름 """
        self.is_greeting_playing = True
        now = datetime.now()
        ampm = "오전" if now.hour < 12 else "오후"
        display_hour = now.hour if now.hour <= 12 else now.hour - 12
        if display_hour == 0:
            display_hour = 12

        # 복약 시간 판단: 일반 복약 시간은 낮 12시, 오후 6시 기준
        medication_hours = [12, 18]

        # 시연 안정성을 위해 정각 전후 10분까지는 복약 시간으로 인정
        self.is_medication_time = any(
            now.hour == h and now.minute <= 10 for h in medication_hours
        )

        # 현재 시각 기준 다음 복약 시간 계산
        next_med_hour = None
        for h in medication_hours:
            if now.hour < h or (now.hour == h and now.minute <= 10):
                next_med_hour = h
                break

        if next_med_hour is None:
            next_med_text = "내일 낮 12시"
        elif next_med_hour == 12:
            next_med_text = "오늘 낮 12시"
        elif next_med_hour == 18:
            next_med_text = "오늘 오후 6시"

        # 🔊 기획안 공동 오프닝 인사 출격
        if not self.is_medication_time:
            self.speak(f"안녕하세요! 저는 젠젠입니다. 현재 시각은 {ampm} {display_hour}시 {now.minute}분입니다. 다음 복약 시간은 {next_med_text}예요. 혹시 지금 몸 상태는 어떠세요?")
            time.sleep(0.8)
        else:
            self.speak("안녕하세요! 저는 젠젠입니다.")
            time.sleep(0.7)
            self.speak(f"현재 시각은 {ampm} {display_hour}시 {now.minute}분입니다.")
            time.sleep(0.8)

        if self.is_medication_time:
            # 🔵 CASE 1 — 복약 시간일 때
            if now.hour == 12:
                medication_label = "점심"
            elif now.hour == 18:
                medication_label = "저녁"
            else:
                medication_label = "정기"

            self.speak(f"지금은 {medication_label} 복약 시간이에요! 오늘 이 시간엔 혈압약과 소화제를 드셔야 해요.")
            time.sleep(4.5)
        else:
            # 🟡 CASE 2 — 복약 시간이 아닐 때 (시연장 대응)
            pass

        if self.is_medication_time:
            self.speak("지금 몸 상태는 어떠세요?")
            time.sleep(0.7)

        self.listen_ready_time = time.time() + 1.0  # 로봇 자기 목소리 잔향/큐에 쌓인 STT 3초 무시
        self.is_greeting_playing = False
        self.response_received = False
        self.get_logger().info("👂 [상태 경청 시작] 1초 후부터 어르신의 답변을 기다립니다...")
        
        # ⏱️ 답변 대기 10초 무응답 타이머 가동
        self.timeout_timer = self.create_timer(10.0, self.handle_timeout)

    def speak(self, text):
        try:
            self.last_spoken_text = text.replace(" ", "")
            self.get_logger().info(f"🔊 말하기: '{text}'")
            tts = gTTS(text=text, lang='ko')
            tts.save("/tmp/final_voice.mp3")
            os.system("ffmpeg -y -i /tmp/final_voice.mp3 -ar 16000 -ac 2 -filter:a 'atempo=1.10,volume=1.6' /tmp/final_voice.wav > /dev/null 2>&1")
            os.system("aplay -D pulse -q /tmp/final_voice.wav")
            self.get_logger().info("🔇 말하기 종료")
        except Exception as e:
            self.get_logger().error(f"❌ 스피커 재생 실패: {e}")


    def play_random_song(self):
        """music 폴더 안의 mp3/wav 중 하나를 60초 재생"""
        try:
            music_dir = "/home/lerobot/aicapstone/lerobot/capstone/gengen/music"
            songs = []
            songs.extend(glob.glob(os.path.join(music_dir, "*.mp3")))
            songs.extend(glob.glob(os.path.join(music_dir, "*.wav")))

            if not songs:
                self.speak("재생할 노래 파일을 찾지 못했어요. music 폴더에 노래 파일을 넣어주세요.")
                return

            song_path = random.choice(songs)
            out_wav = "/tmp/gengen_song.wav"

            self.get_logger().info(f"🎵 노래 재생 시작: {song_path}")

            # TTS와 같은 형식으로 변환: 16000Hz / 2ch
            cmd_convert = f'ffmpeg -y -t 60 -i "{song_path}" -af volume=0.8 -ar 16000 -ac 2 "{out_wav}" > /dev/null 2>&1'
            ret = os.system(cmd_convert)

            if ret != 0 or not os.path.exists(out_wav):
                self.get_logger().error("❌ 노래 변환 실패: /tmp/gengen_song.wav 생성 안 됨")
                self.speak("노래 파일을 재생할 수 없어요.")
                return

            # 일반 TTS와 같은 스피커 장치로 재생
            play_ret = os.system(f'aplay -D pulse -q "{out_wav}"')

            if play_ret != 0:
                self.get_logger().error("❌ 노래 재생 실패: 스피커 출력 실패")
                self.speak("노래를 재생하려고 했지만 스피커 출력에 실패했어요.")
                return

            self.get_logger().info("🎵 노래 재생 종료")
            self.speak("노래 재생이 끝났어요.")

        except Exception as e:
            self.get_logger().error(f"❌ 노래 재생 실패: {e}")

    def speak_beatbox(self):
        """비트박스/랩 전용: 북치기 박치기 구간만 빠르게 재생"""
        try:
            self.get_logger().info("🎤 비트박스 시작")

            intro = "좋아요! 젠젠이의 특별한 비트박스를 시작합니다!"
            beat = "북치기 박치기 북치기 박치기. 북치기 박치기 북치기 박치기. 치키치키 차카차카 북치기 박치기. 둠칫 둠칫 박치기. 북치기 박치기 치키치키 박치기. 젠젠이가 간다 북치기 박치기!"
            outro = "어때요? 젠젠이 랩 실력, 꽤 괜찮지요? 오늘도 기분 좋은 하루 보내세요!"

            # 1) 인트로 normal
            gTTS(text=intro, lang='ko').save("/tmp/beat_intro.mp3")
            os.system("ffmpeg -y -i /tmp/beat_intro.mp3 -ar 16000 -ac 2 -af volume=1.5 /tmp/beat_intro.wav > /dev/null 2>&1")
            os.system("aplay -D pulse -q /tmp/beat_intro.wav")

            # 2) 비트박스 구간 fast
            gTTS(text=beat, lang='ko').save("/tmp/beat_fast.mp3")
            os.system("ffmpeg -y -i /tmp/beat_fast.mp3 -ar 16000 -ac 2 -filter:a 'atempo=1.65,volume=1.9' /tmp/beat_fast.wav > /dev/null 2>&1")
            os.system("aplay -D pulse -q /tmp/beat_fast.wav")

            # 3) 아웃트로 normal
            gTTS(text=outro, lang='ko').save("/tmp/beat_outro.mp3")
            os.system("ffmpeg -y -i /tmp/beat_outro.mp3 -ar 16000 -ac 2 -af volume=1.5 /tmp/beat_outro.wav > /dev/null 2>&1")
            os.system("aplay -D pulse -q /tmp/beat_outro.wav")

            self.get_logger().info("🎤 비트박스 종료")
        except Exception as e:
            self.get_logger().error(f"❌ 비트박스 재생 실패: {e}")


    def reset_to_waiting(self):
        """대화 종료 후 다시 '젠젠아' 호출 대기 상태로 복귀"""
        self.is_awakened = False
        self.is_greeting_playing = False
        self.is_delivering = False
        self.response_received = False
        self.listen_ready_time = time.time() + 1.0
        self.get_logger().info("👂 [대기 복귀] 다시 '젠젠아' 호출을 기다립니다.")


    def check_medication_auto_alert(self):
        """12시/18시에 자동으로 복약 알림을 말함"""
        now = datetime.now()

        # 이미 대화 중이거나 말하는 중이면 자동 알림 방지
        if self.is_awakened or self.is_greeting_playing or self.is_delivering:
            return

        # 12:00~12:01, 18:00~18:01 사이에 한 번만 알림
        if now.hour not in [12, 18] or now.minute != 0:
            return

        alert_key = f"{now.date()}_{now.hour}"
        if self.last_auto_alert_key == alert_key:
            return

        self.last_auto_alert_key = alert_key

        if now.hour == 12:
            meal_label = "점심"
        else:
            meal_label = "저녁"

        self.speak(f"지금은 {meal_label} 약 먹을 시간이에요. 약을 가져다 드릴까요? 필요하시면 젠젠아 하고 불러주세요.")
        self.reset_to_waiting()

    def handle_timeout(self):
        if self.timeout_timer:
            self.timeout_timer.cancel()
        if self.response_received or self.is_delivering:
            return

        self.response_received = True
        self.get_logger().warn("⏳ [무응답 10초 경과] 기획서 무응답 시나리오 실행.")

        if self.is_medication_time:
            robot_talk = "괜찮으세요? 약 가져다 드릴게요."
            self.speak(robot_talk)
            self.send_arm_command("DELIVER_MEDICATION")
        else:
            robot_talk = "괜찮으세요? 필요하시면 언제든 불러주세요."
            self.speak(robot_talk)
            self.get_logger().info("💤 [대기] 비복약 시간 무응답으로 대기 종료.")
            self.reset_to_waiting()

    def send_arm_command(self, command):
        self.is_delivering = True
        arm_msg = String()
        arm_msg.data = str(command).strip()
        self.get_logger().info(f"🚀 로봇 팔 명령 발행: {command}")
        self.arm_pub.publish(arm_msg)

        # 시연 반복을 위해 로봇팔 명령 발행 후 다시 '젠젠아' 호출 대기 상태로 복귀
        time.sleep(1.0)
        self.reset_to_waiting()


    def extract_intent_text(self, normalized_text):
        """긴 STT 문장에서 실제 사용자 의도 키워드만 추출"""
        intent_rules = [
            ("노래", [
                "노래틀어줘", "노래틀어", "노래", "음악틀어줘", "음악틀어", "음악", "뮤직", "틀어줘"
            ]),
            ("비트박스", [
                "비트박스", "비트", "박스", "랩"
            ]),
            ("머리아파", [
                "머리아파", "머리아프", "머리아", "머리", "두통", "지끈", "어지러", "어지", "핑"
            ]),
            ("배아파", [
                "배아파", "배아프", "배아", "배가아파", "배가", "베아파", "베아프",
                "복통", "복부", "속아파", "속아프", "속안좋", "속이안좋",
                "속쓰려", "속쓰림", "소화안돼", "소화안되", "체했", "체한",
                "더부룩", "메스꺼", "토할"
            ]),
            ("피곤", [
                "피곤", "힘들", "지쳐", "고단"
            ]),
            ("좋아", [
                "괜찮", "좋아", "안아파", "멀쩡", "다행"
            ]),
            ("심심", [
                "심심", "그냥", "불러봤"
            ]),
        ]

        for intent, keywords in intent_rules:
            if any(k in normalized_text for k in keywords):
                return intent

        return None

    def stt_callback(self, msg):
        user_text = msg.data.strip()
        normalized_text = user_text.replace(" ", "")
        intent_text = self.extract_intent_text(normalized_text)

        # 로봇이 방금 말한 자기 음성이 STT로 다시 들어온 경우 무시
        # 단, 사용자 명령/증상 키워드가 함께 들어 있으면 사용자 발화로 보고 통과
        user_intent_words = [
            "노래", "노래틀어", "노래틀어줘", "음악", "음악틀어", "음악틀어줘", "뮤직", "틀어줘",
            "랩", "비트박스", "비트", "박스",
            "머리", "머리아파", "머리아프", "두통", "아파", "아프",
            "배", "배아파", "배아프", "속", "속안좋", "소화", "더부룩",
            "피곤", "힘들", "괜찮", "좋아", "안아파"
        ]

        has_user_intent = any(w in normalized_text for w in user_intent_words)

        if self.last_spoken_text:
            echo_hit = (
                normalized_text in self.last_spoken_text
                or self.last_spoken_text in normalized_text
                or any(piece in normalized_text for piece in [
                    "안녕하세요", "저는젠젠", "현재시각", "다음복약시간",
                    "몸상태", "어떠세요", "약먹을시간", "약을가져다드릴까요",
                    "필요하시면", "젠젠아하고", "불러주세요",
                    "다시한번말씀", "잘알아듣지못했어요",
                    "다행이에요", "괜찮으세요"
                ])
            )
            if echo_hit and not has_user_intent:
                self.get_logger().info(f"🔇 [무시] 로봇 자기 말 STT 에코 무시: '{user_text}'")
                return
            elif echo_hit and has_user_intent:
                self.get_logger().info(f"✅ [통과] 로봇 에코가 섞였지만 사용자 명령 포함: '{user_text}'")

        # 로봇이 방금 말한 TTS가 마이크로 다시 들어오는 에코 방지
        # 단, 사용자가 명확한 명령/증상을 말한 경우는 무시 구간이어도 통과
        force_accept_words = [
            "노래", "노", "랩", "비트박스", "비트", "비", "박스", "음악", "뮤직",

            # 증상 명령은 로봇 발화 직후라도 무시하지 않음
            "머리", "머리아파", "머리아프", "머리아픔", "두통", "지끈", "어지러", "어지", "핑",
            "아파", "아프", "아픔", "아픈",
            "배", "배가", "배아", "배아파", "배아프", "배아픔",
            "베", "베아파", "베아프",
            "복통", "복부", "속", "속아파", "속아프", "속안좋", "속이안좋",
            "속쓰려", "속쓰림", "위", "위아파", "소화", "소화안돼", "소화안되",
            "체했", "체한", "더부룩", "미식", "구역", "메스꺼", "토할",
            "피곤", "힘들", "지쳐", "고단"
        ]
        if time.time() < self.listen_ready_time and not any(w in normalized_text for w in force_accept_words):
            self.get_logger().info(f"🔇 [무시] 로봇 발화 직후 STT 에코 무시: '{user_text}'")
            return

        if self.is_greeting_playing or self.is_delivering or self.response_received:
            return
        # wakeword 판정
        # 긴 STT 문장 안에서도 "젠젠아/젠재아/쟁젠아" 같은 강한 호출어가 있으면 호출로 인정
        # 단, "젠" 한 글자만 긴 문장에 섞인 경우는 오작동 방지를 위해 짧은 문장일 때만 인정
        strong_wakewords = [
            "젠젠아", "젠재아", "젠제아", "쟁젠아", "잰젠아",
            "젠젠", "젠재", "젠제", "쟁젠", "잰젠", "잰잰", "쟁쟁", "제제", "재재"
        ]

        fuzzy_wake_chars = ["젠", "잰", "쟁", "제", "재"]
        is_short_wakeword = len(normalized_text) <= 8

        is_wakeword_detected = (
            any(word in normalized_text for word in strong_wakewords)
            or (
                is_short_wakeword
                and any(ch in normalized_text for ch in fuzzy_wake_chars)
            )
        )

        # 1️⃣ [첫 단계] 아직 로봇이 안 깨어났을 때 (선제 호출 대기)
        if not self.is_awakened:
            if is_wakeword_detected:
                self.is_awakened = True
                self.get_logger().info("🔥 [로봇 깨어남] 사용자가 '젠젠아'로 로봇을 호출했습니다!")
                self.trigger_opening_flow()
            return

        # 2️⃣ [두 번째 단계] 로봇이 질문한 후, 사용자의 증상 답변 처리
        if self.timeout_timer:
            self.timeout_timer.cancel()
        self.response_received = True

        if intent_text is not None:
            self.get_logger().info(f"✂️ [의도 추출] 긴 STT 문장에서 핵심 의도만 사용: '{intent_text}' / 원문: '{user_text}'")
            normalized_text = intent_text

        self.get_logger().info(f"📥 답변 수신 및 분류: '{user_text}'")
        self.get_logger().info(f"🔎 정규화 텍스트: '{normalized_text}'")

        # 🚨 최우선 증상 처리: STT가 살짝 깨져도 머리/배 아픔은 먼저 잡기
        if any(w in normalized_text for w in [
            "머리", "머리아", "머리아파", "머리아프", "머리아픔",
            "머리ㅣ", "머리이", "두통", "지끈", "어지러", "어지", "핑"
        ]):
            robot_talk = "아이고 머리가 많이 지끈거리시겠어요. 얼른 타이레놀 약통을 집어서 가져다 드릴게요. 물이랑 같이 꼭 드셔요."
            self.speak(robot_talk)
            self.send_arm_command("DELIVER_TYLENOL")
            return

        if any(w in normalized_text for w in [
            "배", "배가", "배아", "배아파", "배아프", "배아픔",
            "베", "베아파", "베아프",
            "복통", "복부", "복부통증",
            "속", "속아파", "속아프", "속안좋", "속이안좋", "속쓰려", "속쓰림",
            "위", "위아파", "위가아파",
            "소화", "소화안돼", "소화안되", "체했", "체한", "체한것",
            "더부룩", "미식", "구역", "메스꺼", "토할"
        ]):
            robot_talk = "속이 더부룩하시면 정말 괴롭지요. 따뜻한 물 한 잔 준비하시고 계셔요. 제가 소화제 약통 바로 챙겨갈게요."
            self.speak(robot_talk)
            self.send_arm_command("DELIVER_DIGESTIVE")
            return

        # 🎯 예능/돌발 A: 그냥 심심해서 불러봤어 분기
        if any(w in user_text for w in ["심심", "그냥", "불러봤"]):
            robot_talk = "아무 데도 안 아프시니 정말 다행입니다! 오늘 하루도 건강하고 행복하게 보내세요!"
            self.speak(robot_talk)
            self.get_logger().info("💤 [종료] 장난 대답으로 인한 로봇 팔 대기 및 시나리오 완료.")

        # 🎯 예능/돌발 B-1: 노래 틀어줘 / 음악 틀어줘
        elif any(w in normalized_text for w in [
            "노래", "노래틀어", "노래틀어줘", "음악", "음악틀어", "음악틀어줘", "뮤직", "틀어줘"
        ]):
            self.play_random_song()
            self.get_logger().info("🎵 [종료] 노래 재생 완료 후 대기 복귀")
            self.reset_to_waiting()
            return

        # 🎯 예능/돌발 B-2: 랩해줘 / 비트박스 해줘
        elif any(w in normalized_text for w in [
            "랩", "비트박스", "비트", "비", "박스"
        ]):
            self.speak_beatbox()
            self.get_logger().info("🎵 [종료] 비트박스 서비스 완료 후 로봇 팔 대기 및 시나리오 완료.")
            self.reset_to_waiting()
            return

        # A. 머리 아픔 / 어지러움
        elif any(w in normalized_text for w in [
            "머리", "머리아", "머리아파", "머리아픔", "머리아프",
            "두통", "아파", "아프", "아픈", "지끈", "어지러", "어지", "핑"
        ]):
            robot_talk = "아이고 머리가 많이 지끈거리시겠어요. 얼른 타이레놀 약통을 집어서 가져다 드릴게요. 물이랑 같이 꼭 드셔요."
            self.speak(robot_talk)
            self.send_arm_command("DELIVER_TYLENOL")

        # B. 배 아픔 / 속 안 좋음
        elif any(w in normalized_text for w in [
            "배", "배가", "배아", "배아파", "배아프", "배아픔",
            "베", "베아파", "베아프",
            "복통", "복부", "복부통증",
            "속", "속아파", "속아프", "속안좋", "속이안좋", "속쓰려", "속쓰림",
            "위", "위아파", "위가아파",
            "소화", "소화안돼", "소화안되", "체했", "체한",
            "미식", "구역", "더부룩", "메스꺼", "토할"
        ]):
            robot_talk = "속이 더부룩하시면 정말 괴롭지요. 따뜻한 물 한 잔 준비하시고 계셔요. 제가 소화제 약통 바로 챙겨갈게요."
            self.speak(robot_talk)
            self.send_arm_command("DELIVER_DIGESTIVE")

        # C. 피곤해 / 힘들해 (CASE 2 단독)
        elif any(w in normalized_text for w in ["피곤", "힘들", "지쳐", "고단"]):
            robot_talk = "많이 피곤하시군요. 오늘은 무리하지 마시고 잠시 쉬어가셔도 괜찮아요. 필요하시면 언제든 젠젠을 불러주세요."
            self.speak(robot_talk)
            self.reset_to_waiting()
            return

        # D. 긍정 유형: 괜찮아 / 좋아
        elif any(w in normalized_text for w in ["괜찮", "좋아", "안아파", "멀쩡", "다행"]):
            if self.is_medication_time:
                robot_talk = "다행이에요! 약 가져다 드릴게요."
                self.speak(robot_talk)
                self.send_arm_command("DELIVER_MEDICATION")
            else:
                robot_talk = "다행이에요. 다음 복약 시간에 다시 알려드릴게요!"
                self.speak(robot_talk)
                self.reset_to_waiting()
            return

        # E. 예외 방어
        else:
            if self.is_medication_time:
                robot_talk = "약 드실 시간인 만큼 챙겨드릴게요. 약 바로 가져다 드리겠습니다."
                self.speak(robot_talk)
                self.send_arm_command("DELIVER_MEDICATION")
            else:
                robot_talk = "잘 알아듣지 못했어요. 불편하시면 다시 한 번 말씀해 주세요. 필요하시면 언제든 젠젠을 불러주세요."
                self.speak(robot_talk)
                self.reset_to_waiting()
                return

def main(args=None):
    rclpy.init(args=args)
    node = DialogControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
