import os
import requests

APP_NAME = "IMAGE-AI-SERVICE"
#INSTANCE_PORT = 8000
INSTANCE_PORT = int(os.getenv("INSTANCE_PORT", 8000))
EUREKA_SERVER_URL = "http://52.79.151.83:3150/eureka/"

def get_external_ip():
    try:
        ip = requests.get('https://api.ipify.org').text
        print(f"퍼블릭 IP 주소: {ip}")
        return ip
    except Exception as e:
        print(f"퍼블릭 IP 주소 조회 실패: {e}")

