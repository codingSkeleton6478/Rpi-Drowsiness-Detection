import sqlite3
import datetime

class DrivingLogDB:
    def __init__(self, db_path="driving_log.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """DB 테이블 초기화 (없으면 생성)"""
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
        """이벤트 기록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO drowsiness_events (event_type) VALUES (?)", (event_type,))
                conn.commit()
                # print(f"[DB Log] Event '{event_type}' saved.") 
        except Exception as e:
            print(f"[DB Error] Logging failed: {e}")

    def get_count_last_minutes(self, minutes):
        """최근 N분간 발생한 이벤트 횟수 반환"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
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