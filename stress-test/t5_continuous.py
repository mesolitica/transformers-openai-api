from locust import HttpUser, task
from locust import events
import itertools
import gpustat
import time

gpu_stats = gpustat.GPUStatCollection.new_query()

"""
Make sure already running this,

python3 -m transformers_openai.main \
--host 0.0.0.0 --port 7088 \
--attn-implementation sdpa \
--model-type transformers_openai.models.T5ForConditionalGeneration \
--tokenizer-type AutoTokenizer --tokenizer-use-fast false \
--architecture-type encoder-decoder --torch-dtype bfloat16 \
--cache-type none --continuous-batching true --hf-model google/flan-t5-base
"""

questions = [
    'What is the capital of Peru?',
    'Do you prefer sweet or salty snacks?',
    'What is the largest planet in our solar system?',
    'Have you ever been to a music festival?',
    'What is the chemical symbol for gold?',
    'Can you speak more than one language?',
    'What is the highest mountain peak in North America?',
    'Do you have a pet?',
    'What is the smallest country in the world?',
    'Have you ever gone skydiving?',
    'What is the most popular social media platform?',
    'Can you play a musical instrument?',
    'What is the largest living species of lizard?',
    'Have you ever been on a cruise?',
    'What is the deepest ocean trench?',
    'Do you prefer reading books or watching movies?',
    'What is the fastest land animal?',
    'Have you ever tried bungee jumping?',
    'What is the most widely spoken language in the world?',
    "Can you solve a Rubik's Cube?",
    'What is the highest recorded temperature on Earth?',
    'Do you prefer summer or winter?',
    'What is the largest mammal on Earth?',
    'Have you ever gone on a hot air balloon ride?',
    'What is the longest river in South America?',
    'Can you recite the alphabet backwards?',
    'What is the highest mountain peak in Europe?',
    'Do you have a favorite sports team?',
    'What is the chemical symbol for silver?',
    'Have you ever gone scuba diving?',
    'What is the largest city in Asia?',
    'Can you speak in front of a large crowd?',
    'What is the deepest lake in the world?',
    'Do you prefer coffee or tea?',
    'What is the highest mountain peak in Africa?',
    'Have you ever been to a comedy club?',
    'What is the smallest bone in the human body?',
    'Can you ride a unicycle?',
    'What is the longest word in the English language?',
    'Do you prefer cats or dogs?',
    'What is the highest recorded wind speed?',
    'Have you ever gone ziplining?',
    'What is the largest waterfall in the world?',
    'Can you solve a Sudoku puzzle?',
    'What is the highest mountain peak in Australia?',
    'Do you prefer hiking or biking?',
    'What is the largest island in the Mediterranean?',
    'Have you ever gone snorkeling?',
    'What is the highest mountain peak in South America?',
    'Can you touch your nose with your tongue?'
]

questions = itertools.cycle(questions)


class HelloWorldUser(HttpUser):

    host = "http://127.0.0.1:7088"

    @task
    def hello_world(self):

        json_data = {
            'model': 'model',
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 50,
            'max_tokens': 256,
            'truncate': 2048,
            'repetition_penalty': 1,
            'stop': [],
            'messages': [
                {
                    'role': 'user',
                    'content': f'Q: {next(questions)}</s>',
                },
            ],
            'stream': False,
        }
        r = self.client.post('/chat/completions', json=json_data)
