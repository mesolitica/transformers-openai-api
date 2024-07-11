from pydantic import BaseModel
from typing import Union, List, Optional


class Parameters(BaseModel):
    model: str = 'model'
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 256
    truncate: int = 2048
    repetition_penalty: float = 1.0
    stop: List[str] = []


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

    def __str__(self) -> str:
        if self.role == 'system':
            return f'system:\n{self.content}\n'

        elif self.role == 'user':
            if self.content is None:
                return 'user:\n</s>'
            else:
                return f'user:\n</s>{self.content}\n'

        elif self.role == 'assistant':

            if self.content is None:
                return 'assistant'

            else:
                return f'assistant:\n{self.content}\n'

        else:
            raise ValueError(f'Unsupported role: {self.role}')


class ChatCompletionForm(Parameters):
    messages: List[ChatMessage]
    stream: bool = False
