import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool
import os

# Load environment variables
load_dotenv()

# Define structured response model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools
tools = [search_tool]

# Agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="AI Research Chatbot", page_icon="🤖")

st.title("🤖 AI Research Assistant")
st.write("Ask me anything and I’ll research it for you!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Type your question here..."):
    # Add user query to chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            raw_response = agent_executor.invoke({"query": query})
            try:
                structured_response = parser.parse(raw_response.get("output"))
                response_text = f"**Topic:** {structured_response.topic}\n\n**Summary:** {structured_response.summary}\n\n**Sources:** {', '.join(structured_response.sources)}\n\n**Tools Used:** {', '.join(structured_response.tools_used)}"
            except Exception:
                response_text = raw_response.get("output", "Sorry, I couldn’t parse the response.")

            st.markdown(response_text)

    # Save response in history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
