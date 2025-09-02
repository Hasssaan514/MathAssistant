from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from tools import search_tool
# Load environment variables
load_dotenv()

#your_api_key_here = os.getenv("")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

your_api_key_here = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=your_api_key_here  # 👈 pass API key directly
)

#response = llm.invoke("Hello Gemini, how are you?")
#print(response.content)

parser =PydanticOutputParser(pydantic_object= ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and user necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools= [search_tool]
agent= create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]

)

agent_executor= AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What is your query? ")
raw_response = agent_executor.invoke({"query": query})
print("Raw response:", raw_response)

structured_response = parser.parse(raw_response.get("output"))
print("Parsed response:", structured_response)


# Run a test
