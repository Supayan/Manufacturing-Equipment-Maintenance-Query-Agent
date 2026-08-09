from dataclasses import dataclass
from typing import Protocol
import os
import openai
import time
import ollama

@dataclass
class LLMResponse:
    text:str
    latency_ms:int
    model:str

class LLMProvider(Protocol):
    def complete(self, system:str, user:str) -> LLMResponse: ...
    
class OpenAIProvider:
    def __init__(self,settings):
        self.model=settings.generation.model
        self.temperature = settings.generation.temperature
        self.max_token = settings.generation.max_token
        self.api_key = os.environ["OPEN_API_KEY"]
        
    def complete(self,system:str, user:str)-> LLMResponse:
        openai.api_key= self.api_key
        start = time.perf_counter()
        response = openai.ChatCompletion.create(
            model=self.model,
            message=[{"role":"system","content":system},
                     {"role":"user","content":user}],
            temperature=self.temperature,
            max_token=self.max_token
        )
        end=time.perf_counter()
        text=response["choices"][0]["message"]["content"]
        latency = end-start
        return LLMResponse(text=text,latency_ms=latency,model=self.model)


class OllamaProvider:
    def __init__(self,settings):
        self.model=settings.generation.model
        self.temperature = settings.generation.temperature
        self.max_token = settings.generation.max_token
    
    def complete(self, system:str, user:str)-> LLMResponse:
        start = time.perf_counter()
        response = ollama.chat(
            model=self.model,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_token,
            }
        )
        end = time.perf_counter()
        text = response["message"]["content"]
        latency = end - start
        return LLMResponse(text=text, latency_ms=latency, model=self.model)
    
