import sqlite3
import datetime

class DrivingLogDB:
    """
    [운행 기록 데이터베이스 관리 클래스]
    - 역할: 운전 중 발생하는 주요 이벤트(졸음, 비상 상황)를 영구 저장소(SQLite)에 기록
    - 목적: 단순 실시간 감시를 넘어, 운전 종료 후 리포트를 제공하거나 
            장시간 운행 시 '누적 피로도'를 기반으로 경고하기 위함.
    """
    def __init__(self, db_path="driving_log.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """
        [DB 초기화]
        - 앱 실행 시 테이블 존재 여부를 확인하고 없으면 생성함.
        - Schema:
            id: 고유 식별자 (Auto Increment)
            timestamp: 이벤트 발생 시간 (자동 입력)
            event_type: 이벤트 종류 (예: SLEEP, DROWSY, ALARM)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS drowsiness_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"[DB Error] Init failed: {e}")

    def log_event(self, event_type="SLEEP"):
        """
        [이벤트 기록]
        - Analyzer가 졸음 카운트 증가(Rising Edge)를 감지했을 때 호출됨.
        - 매 프레임 기록하는 것이 아니라 '사건' 단위로 기록하여 DB 부하를 최소화함.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO drowsiness_events (event_type) VALUES (?)", (event_type,))
                conn.commit()
                # print(f"[DB Log] Event '{event_type}' saved.") # 디버깅용 출력 (주석 처리됨)
        except Exception as e:
            print(f"[DB Error] Logging failed: {e}")

    def get_count_last_minutes(self, minutes):
        """
        [통계 쿼리: 최근 N분간 졸음 횟수 조회]
        - 용도: 주기적 리포트(30분마다 알림) 기능에서 사용.
        - SQL Logic: 현재 시간(datetime('now'))에서 N분을 뺀 시점 이후의 데이터 개수(Count)를 반환.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # SQLite의 datetime 함수를 사용하여 시간 차이를 계산
                query = f"""
                    SELECT COUNT(*) FROM drowsiness_events 
                    WHERE timestamp >= datetime('now', '-{minutes} minutes')
                """
                cursor.execute(query)
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            print(f"[DB Error] Count fetch failed: {e}")
            return 0