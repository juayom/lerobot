import pyaudio

def main():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("\n========================================================")
    print("🎤 현재 젯슨 보드 사운드 장치 스캔 결과 (인덱스 번호 확인)")
    print("========================================================")
    
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        # 입력(마이크) 채널이 1개 이상인 장치들만 필터링해서 출력
        if device_info.get('maxInputChannels') > 0:
            print(f"▶ [장치 번호: {i}] {device_info.get('name')}")
            print(f"   - 최대 입력 채널 수: {device_info.get('maxInputChannels')}\n")
            
    print("========================================================\n")
    p.terminate()

if __name__ == "__main__":
    main()
