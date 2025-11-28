import wave
import numpy as np  # audioop 대신 numpy 사용

FILENAME = "user_input.wav"  # 혹은 broadcast.wav

try:
    with wave.open(FILENAME, 'rb') as wf:
        # 오디오 데이터 읽기
        raw_data = wf.readframes(wf.getnframes())
        
        # 바이트 데이터를 numpy 배열(int16)로 변환
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        
        # 1. RMS (Root Mean Square) 계산
        # (제곱 -> 평균 -> 제곱근)
        if len(audio_data) == 0:
            rms = 0
            max_val = 0
        else:
            rms = int(np.sqrt(np.mean(audio_data**2)))
            max_val = np.max(np.abs(audio_data))

        print("-" * 30)
        print(f"📊 [노이즈 분석 결과 (NumPy 버전)]")
        print(f"파일 이름: {FILENAME}")
        print(f"평균 노이즈 크기 (RMS): {rms}")
        print(f"최대 피크 크기 (Max): {max_val}")
        print("-" * 30)
        print(f"💡 추천 설정값: {int(rms * 1.5)}") 
        print("(이 값보다 작은 소리는 코드가 무시하게 됩니다)")
        print("-" * 30)

except FileNotFoundError:
    print(f"❌ '{FILENAME}' 파일을 찾을 수 없습니다. 녹음을 한 번 실행한 뒤 다시 시도해주세요.")
except Exception as e:
    print(f"오류 발생: {e}")