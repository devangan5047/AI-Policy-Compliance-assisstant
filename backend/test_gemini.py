import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = ChatOpenRouter(
    api_key=api_key,
    model="openrouter/free",
    temperature=0.7,
)

response = client.invoke("what is machine learning?")
print(response.content)