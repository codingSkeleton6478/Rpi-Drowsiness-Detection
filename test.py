import sqlite3

def view_logs():
    db_path = "driving_log.db"
    print(f"📂 [{db_path}] 데이터 조회 중...\n")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 최근 20개 데이터 조회
        cursor.execute("SELECT * FROM drowsiness_events ORDER BY id DESC LIMIT 20")
        rows = cursor.fetchall()
        
        if not rows:
            print("📭 데이터가 없습니다. (아직 졸음 이벤트가 감지되지 않았거나, main.py가 실행되지 않음)")
        else:
            print(f"{'ID':<5} | {'TIMESTAMP':<20} | {'EVENT TYPE'}")
            print("-" * 40)
            for row in rows:
                # row[0]: id, row[1]: timestamp, row[2]: event_type
                print(f"{row[0]:<5} | {row[1]:<20} | {row[2]}")
            
            print("-" * 40)
            print(f"총 {len(rows)}개의 최근 기록을 표시했습니다.")

    except sqlite3.OperationalError:
        print("❌ 오류: DB 파일(driving_log.db)을 찾을 수 없습니다. main.py를 먼저 실행해 주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    view_logs()