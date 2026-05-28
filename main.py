from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily=TavilyClient()

@tool
def search(query:str)->str:
    """
    Tool that searches the internet
    Args:
        query: The query to search
    Returns:
        The search result
    """
    print(f"Query -> {query}")
    return tavily.search(query=query)

llm = ChatGroq(temperature=0, model="meta-llama/llama-4-scout-17b-16e-instruct")
tools=[search]
agent=create_agent(model=llm,tools=tools)
def main():
    response = agent.invoke({"messages":HumanMessage(content="Search and find the latest REITS news today")})
    print(response)
    response2 = agent.invoke({"messages":HumanMessage(content="Search and find the latest jobs on AI Engineer in Hyderabad")})
    print(response2)

if __name__ == "__main__":
    main()