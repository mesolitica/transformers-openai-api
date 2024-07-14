from locust import HttpUser, task
from locust import events
import itertools
import gpustat
import time

gpu_stats = gpustat.GPUStatCollection.new_query()

"""
Make sure already running this,

CUDA_VISIBLE_DEVICES=0 HF_TRANSFER=1 \
python3.10 -m transformers_openai.main \
--host 0.0.0.0 --port 7088 \
--attn-implementation sdpa \
--model-type WhisperForConditionalGeneration \
--processor-type AutoProcessor \
--serving-type whisper --torch-dtype bfloat16 --cache-type none \
--hf-model openai/whisper-large-v3
"""


class HelloWorldUser(HttpUser):

    host = "http://127.0.0.1:7088"

    @task
    def hello_world(self):

        headers = {
            'accept': 'application/json',
        }

        files = {
            'top_k': (None, '50'),
            'timestamp_granularities': (None, 'segment'),
            'top_p': (None, '0.95'),
            'model': (None, 'base'),
            'temperature': (None, '0'),
            'response_format': (None, 'verbose_json'),
            'language': (None, 'ms'),
            'repetition_penalty': (None, '1'),
            'file': ('test.mp3', open('audio/7021-79759-0004.wav', 'rb'), 'audio/mpeg'),
        }
        r = self.client.post('/audio/transcriptions', headers=headers, files=files)
