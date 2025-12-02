import os
import subprocess
from pyngrok import ngrok

# 경로 이동
PROJECT_PATH = '/Users/choihj/PycharmProjects/FirstFin'
os.chdir(PROJECT_PATH)
print(f"📁 이동 완료: {os.getcwd()}")

# 새로운 포트 사용 (8502 사용)
port = 8502

# ngrok 새 터널 실행
public_url = ngrok.connect(port).public_url
print(f"\n🌐 접속 주소: {public_url}")

# Streamlit 실행
process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", str(port), "--server.headless", "true"],
    stdout=open("streamlit.log", "w"),
    stderr=subprocess.STDOUT
)
print(f"✅ Streamlit 실행 중 (PID: {process.pid})")

'''
사용방법
cd /Users/choihj/PycharmProjects/FirstFin && pkill -9 -f streamlit; pkill -9 -f ngrok; rm -rf ~/.streamlit/cache .streamlit/cache __pycache__ cache .pytest_cache finz_memory_*.txt streamlit.log; sleep 2; streamlit run app.py --server.port 8502
'''
