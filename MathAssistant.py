import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
import sympy as sp
import matplotlib.pyplot as plt
import os

# Load environment variables
load_dotenv()


# ----------------- TOOL: Sympy Solver -----------------
@tool
def solve_equation(equation: str) -> str:
    """
    Solve a math equation (algebra/trigonometry/etc.) and return step-by-step solution.
    Example input: "solve x^2 - 5x + 6 = 0"
    """
    try:
        x = sp.symbols("x")
        steps = []
        if "=" in equation:
            expr = sp.sympify(equation.split("=")[0]) - sp.sympify(equation.split("=")[1])
        else:
            expr = sp.sympify(equation)

        sol = sp.solve(expr, x)
        steps.append(f"Equation: ${sp.latex(expr)} = 0$")
        steps.append(f"Simplified form: ${sp.latex(sp.simplify(expr))}$")
        steps.append(f"Solutions: {', '.join([sp.latex(s) for s in sol])}")
        return "\n".join([str(s) for s in steps])
    except Exception as e:
        return f"Error solving equation: {str(e)}"


# ----------------- Structured Response -----------------
class MathResponse(BaseModel):
    problem: str
    solution_steps: str
    final_answer: str


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Parser
parser = PydanticOutputParser(pydantic_object=MathResponse)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a **Math Tutor Assistant**.
            Solve equations step by step (algebra, trigonometry, geometry, school-level).
            Use tools when needed. Always explain clearly like a teacher.
            Wrap your answer only in this format:\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools
tools = [solve_equation]

# Agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Math Assistant Chatbot", page_icon="🧮")

st.title("🧮 Math Assistant Chatbot")
st.write("Ask me any school-level Math problem and I’ll solve step by step!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Type your math problem here... (e.g. solve x^2 - 5x + 6 = 0)"):
    # Add user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            raw_response = agent_executor.invoke({"query": query})
            try:
                structured_response = parser.parse(raw_response.get("output"))

                # Render LaTeX properly
                st.markdown(f"**Problem:** {structured_response.problem}")
                st.markdown("**Steps:**")
                for line in structured_response.solution_steps.split("\n"):
                    if line.strip().startswith("$"):
                        st.latex(line.strip("$"))
                    else:
                        st.write(line)
                st.markdown("**Final Answer:**")
                st.latex(structured_response.final_answer)

                # Try plotting if it looks like a function
                try:
                    x = sp.Symbol("x")
                    expr = sp.sympify(query.replace("solve", "").strip())
                    fig, ax = plt.subplots()
                    sp.plot(expr, (x, -10, 10), show=False).save("temp_plot.png")
                    st.image("temp_plot.png", caption="Graph of the function")
                except Exception:
                    pass

                response_text = f"Problem: {structured_response.problem} | Final Answer: {structured_response.final_answer}"
            except Exception:
                response_text = raw_response.get("output", "Sorry, I couldn’t parse the response.")

    # Save in history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
