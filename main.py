from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def main():
    information="""
    Real Estate Investment Trusts (REITs) are companies that own, operate, or finance income-generating real estate. Modeled after mutual funds, they pool capital from multiple investors. This allows individuals to earn a share of rental income and capital appreciation from commercial properties without having to buy or manage physical buildings.
    """

    prompt=f"""
    given the information {information} how should a person add certain allocation into his portfolio?
    """

    summary_prompt_template=PromptTemplate(input_variables=["information"],template=prompt)

    llm = ChatGroq(temperature=0,model="llama-3.3-70b-versatile")

    chain =summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.content)


if __name__ == "__main__":
    main()