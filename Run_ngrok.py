import os
import time
import subprocess
from pyngrok import ngrok
from pyngrok import conf

# ----------------------------------------
# 1) 이전 프로세스/캐시 정리
# ----------------------------------------
os.system("pkill -9 -f ngrok")
os.system("pkill -9 -f streamlit")
os.system("rm -rf ~/.ngrok2")
os.system("rm -rf ~/.config/ngrok")

# ----------------------------------------
# 3) 프로젝트 경로 이동
# ----------------------------------------
PROJECT_PATH = "/Users/choihj/PycharmProjects/FirstFin"
os.chdir(PROJECT_PATH)
print(f"📁 이동 완료: {os.getcwd()}")

port = 8502

# ----------------------------------------
# 4) Streamlit 먼저 실행 (중요)
# ----------------------------------------
process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", str(port), "--server.headless", "true"],
    stdout=open("streamlit.log", "w"),
    stderr=subprocess.STDOUT
)

print(f"🚀 Streamlit 실행됨 (PID {process.pid})")

# Streamlit 완전히 뜰 때까지 기다려줌
time.sleep(4)

# ----------------------------------------
# 5) ngrok 터널 생성 (Streamlit 이후 실행)
# ----------------------------------------
public_url = ngrok.connect(addr=port, proto="http").public_url
print(f"\n🌐 친구에게 보낼 주소: {public_url}\n")

'''
사용 방법
pkill -9 -f ngrok
ngrok http 8502
'''