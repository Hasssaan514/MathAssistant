import re
import sympy as sp
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import matplotlib.pyplot as plt
import os
from PIL import Image
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# ----------------- STEP 1: Equation Cleaner -----------------
def clean_equation(equation: str) -> str:
    """Convert messy copied equations into LaTeX + Sympy-friendly input"""
    # Handle HTML-like subscripts and superscripts
    equation = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", equation)
    equation = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", equation)
    # Remove stray tags
    equation = re.sub(r"<.*?>", "", equation)

    # Replace common math words/symbols
    equation = equation.replace("sec", "1/cos")
    equation = equation.replace("csc", "1/sin")
    equation = equation.replace("cot", "cos/sin")
    equation = equation.replace("²", "**2")

    return equation.strip()

# ----------------- TOOL: Sympy Solver (Step 3 uses cleaned input) -----------------
@tool
def solve_equation(equation: str) -> str:
    """
    Solve a math equation (algebra/trigonometry/etc.) and return step-by-step solution.
    Example input: "solve x^2 - 5x + 6 = 0"
    """
    try:
        x = sp.symbols("x")
        steps = []

        # Clean input before using Sympy
        cleaned = clean_equation(equation)

        if "=" in cleaned:
            expr = sp.sympify(cleaned.split("=")[0]) - sp.sympify(cleaned.split("=")[1])
        else:
            expr = sp.sympify(cleaned)

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

# Initialize LLM (multimodal)
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
            If given an image, extract the math problem first.
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
st.write("Type your math problem OR upload a screenshot, and I’ll solve step by step!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- IMAGE HANDLING ----------------
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

uploaded_file = st.file_uploader("📸 Upload a screenshot of your math problem", type=["png", "jpg", "jpeg"])
query = None

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Problem", use_column_width=True)

    st.info("⏳ Reading equation from image...")
    img_base64 = encode_image_to_base64(img)

    vision_response = llm.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract the math equation from this image and return in plain text format."},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
            ]
        )
    ])
    query = vision_response.content.strip()
    st.success(f"✅ Extracted Equation: {query}")

# ---------------- CHAT HANDLING ----------------
text_query = st.chat_input("Type your math problem here... (e.g. solve x^2 - 5x + 6 = 0)")
if text_query:
    query = text_query

if query:
    # Step 2: Show original + LaTeX preview
    st.write("**Original Input:**", query)
    st.write("**LaTeX Preview:**")
    st.latex(clean_equation(query))

    # Save user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Solve with agent
    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            raw_response = agent_executor.invoke({"query": query})
            try:
                structured_response = parser.parse(raw_response.get("output"))

                # Problem statement
                st.markdown(f"**Problem:** {structured_response.problem}")

                # Step-by-step
                st.markdown("**Steps:**")
                for line in structured_response.solution_steps.split("\n"):
                    if line.strip().startswith("$"):
                        st.latex(line.strip("$"))
                    else:
                        st.write(line)

                # Final Answer
                st.markdown("**Final Answer:**")
                st.latex(structured_response.final_answer)

                # Try plotting
                try:
                    x = sp.Symbol("x")
                    expr = sp.sympify(clean_equation(query.replace("solve", "").strip()))
                    fig, ax = plt.subplots()
                    sp.plot(expr, (x, -10, 10), show=False).save("temp_plot.png")
                    st.image("temp_plot.png", caption="Graph of the function")
                except Exception:
                    pass

                response_text = f"Problem: {structured_response.problem} | Final Answer: {structured_response.final_answer}"
            except Exception:
                response_text = raw_response.get("output", "Sorry, I couldn’t parse the response.")

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
