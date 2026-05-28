from dotenv import load_dotenv
import ssl
import httpx
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

http_client = httpx.Client(verify=False)

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile",http_client=http_client,)


def main():
    print("Hello from langchain!")
    llm_response = llm.invoke("What is the capital of France?")
    print(llm_response.content)


if __name__ == "__main__":
    main()
