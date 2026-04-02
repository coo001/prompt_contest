import os
import sys
import pandas as pd
import requests
import math
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

# prompt.txt에서 시스템 프롬프트 로드
with open("prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# 로그 파일 설정
os.makedirs("logs", exist_ok=True)
log_path = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class Tee:
    """터미널과 파일에 동시에 출력"""
    def __init__(self, filepath):
        self.terminal = sys.__stdout__
        self.file = open(filepath, "w", encoding="utf-8")
    def write(self, msg):
        try:
            self.terminal.write(msg)
        except UnicodeEncodeError:
            self.terminal.write(msg.encode(self.terminal.encoding, errors='replace').decode(self.terminal.encoding))
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()

tee = Tee(log_path)
sys.stdout = tee

def classify(title, content):
    user_msg = f"제목: {title}\n본문: {str(content)[:300]}"
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 5,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
            },
            timeout=15
        )
        data = resp.json()

        if resp.status_code != 200 or "error" in data:
            msg = data.get("error", {}).get("message", str(data))
            print(f"  ⚠️  API 오류 ({resp.status_code}): {msg}")
            if resp.status_code == 400:
                return -2  # 스킵 가능한 오류
            return -1

        result = data["choices"][0]["message"]["content"].strip()
        return int(result[0]) if result[0] in "01" else -1

    except Exception as e:
        print(f"  ⚠️  예외 발생: {e}")
        return -1

# 샘플 로드
df = pd.read_csv("cs_data.csv")

print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"프롬프트 파일: prompt.txt ({len(SYSTEM_PROMPT)}자)")
print(f"테스트 시작: {len(df)}개 샘플\n")

results = []
for i, row in df.iterrows():
    pred = classify(row['title'], row['content'])
    if pred == -1:
        print("  → API 키 확인 후 재실행하세요.")
        break
    if pred == -2:
        print(f"  → [{i+1:02d}] 스킵 (400 오류)")
        continue
    correct = pred == row['soft_label']
    results.append({
        "제목": row['title'],
        "정답": row['soft_label'],
        "예측": pred,
        "정오": "O" if correct else "X"
    })
    print(f"[{i+1:02d}] {'O' if correct else 'X'} 정답:{row['soft_label']} 예측:{pred} | {row['title'][:35]}")
    time.sleep(0.3)

if not results:
    print("\n결과 없음. API 키를 확인하세요.")
else:
    res_df = pd.DataFrame(results)
    accuracy = (res_df['정답'] == res_df['예측']).mean()
    L = len(SYSTEM_PROMPT)
    b_score = 0.1 * math.sqrt(1 - (L / 1200) ** 2)
    total_score = 0.9 * accuracy + b_score

    print(f"\n{'='*50}")
    print(f"정확도:       {accuracy:.4f} ({int(accuracy*len(res_df))}/{len(res_df)})")
    print(f"프롬프트 길이: {L}자")
    print(f"B점수:        {b_score:.4f}")
    print(f"예상 총점:    {total_score:.4f}")
    print(f"{'='*50}")

    wrong = res_df[res_df['정답'] != res_df['예측']]
    if len(wrong):
        print(f"\nX 오답 {len(wrong)}개:")
        for _, r in wrong.iterrows():
            print(f"  정답:{r['정답']} 예측:{r['예측']} | {r['제목']}")

    res_df.to_csv("test_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n결과 저장 완료: test_results.csv")
    print(f"로그 저장 완료: {log_path}")

tee.close()
sys.stdout = sys.__stdout__
