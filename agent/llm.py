from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)
