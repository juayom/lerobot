import os
from gtts import gTTS

def main():
    text = "아이고 머리가 많이 지끈거리시겠어요. 얼른 타이레놀 약통을 집어서 가져다 드릴게요. 물이랑 같이 꼭 드셔요."
    print(f"🔊 [하드웨어 다이렉트 쏴주기] 시작: '{text}'")
    
    # mp3 파일 생성 후 저장
    tts = gTTS(text=text, lang='ko')
    tts.save("/tmp/robot_voice.mp3")
    
    # mp3를 ReSpeaker 표준 규격(16kHz, 2채널 wav)으로 초고속 변환
    os.system("ffmpeg -y -i /tmp/robot_voice.mp3 -ar 16000 -ac 2 /tmp/robot_voice.wav > /dev/null 2>&1")
    
    # 💡 우분투 순정 플레이어로 사운드 카드(sysdefault)에 강제 출력!
    os.system("aplay -D sysdefault /tmp/robot_voice.wav")
    print("🔇 [하드웨어 다이렉트] 재생 완료")

if __name__ == "__main__":
    main()
