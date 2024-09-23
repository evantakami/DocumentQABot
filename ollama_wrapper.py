import requests
import json
import time
import logging

logger = logging.getLogger(__name__)

class OllamaChatWrapper:
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model_name = model_name

    def generate(self, prompt, max_retries=3, delay=1):
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model_name, "prompt": prompt},
                    stream=True,
                    timeout=300
                )
                response.raise_for_status()
                
                full_response = ""
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        try:
                            decoded_chunk = chunk.decode('utf-8')
                            if decoded_chunk.strip():
                                json_response = json.loads(decoded_chunk)
                                if 'response' in json_response:
                                    full_response += json_response['response']
                                else:
                                    logger.warning(f"予期しない応答形式: {json_response}")
                        except json.JSONDecodeError:
                            logger.warning(f"JSONデコードエラー: {decoded_chunk}")
                
                return full_response
            except requests.RequestException as e:
                logger.warning(f"リクエストエラー（試行 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    logger.error("最大リトライ回数に達しました。")
                    raise