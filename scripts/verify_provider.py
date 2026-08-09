from app.generation.providers import OpenAIProvider,OllamaProvider
from app.config import settings

#OpenAIProvider(settings).complete("You are john","Say hello!")
response = OllamaProvider(settings).complete("Your name is Supayan Das","What is your name?")

print(response.text)
print(response.model)
print(response.latency_ms)
