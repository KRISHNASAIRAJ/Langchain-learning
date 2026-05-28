from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """A source of information."""
    name: str = Field(..., description="The name of the source.")
    url: str = Field(..., description="The URL of the source.")

class AgentResponse(BaseModel):
    """The response from the agent."""
    content: str = Field(..., description="The content of the response.")
    sources: List[Source] = Field(..., description="The sources of the information.")

llm = ChatGroq(temperature=0, model="meta-llama/llama-4-scout-17b-16e-instruct")
tools=[TavilySearch()]
agent=create_agent(model=llm,tools=tools,)

def main():
    response = agent.invoke({"messages":HumanMessage(content="Search and find the latest REITS news today")})
    print(response)
    response2 = agent.invoke({"messages":HumanMessage(content="Search and find the latest jobs on AI Engineer in Hyderabad")})
    print(response2)

if __name__ == "__main__":
    main()