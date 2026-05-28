from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

def main():
    llm_response = llm.invoke("What are the listed REITS in India?")
    print(llm_response.content)


if __name__ == "__main__":
    main()