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
import os
from PIL import Image
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import plotly.express as px

# Load environment variables
load_dotenv()

# ----------------- STEP 1: Equation Cleaner -----------------
def clean_equation(equation: str) -> str:
    """Convert messy copied equations into LaTeX + Sympy-friendly input"""
    equation = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", equation)
    equation = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", equation)
    equation = re.sub(r"<.*?>", "", equation)
    equation = equation.replace("sec", "1/cos")
    equation = equation.replace("csc", "1/sin")
    equation = equation.replace("cot", "cos/sin")
    equation = equation.replace("²", "**2")
    return equation.strip()

# ----------------- TOOL: Sympy Solver -----------------
@tool
def solve_equation(equation: str) -> str:
    """Solve algebraic or trigonometric equations step by step."""
    try:
        x = sp.symbols("x")
        steps = []
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

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=MathResponse)

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

tools = [solve_equation]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------- PDF Export -----------------
def save_solution_as_pdf(problem, steps, final_answer, filename="solution.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Math Assistant Solution")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "Problem:")
    c.setFont("Courier", 11)
    c.drawString(100, height - 120, problem)

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 160, "Step-by-step Solution:")
    text = c.beginText(70, height - 180)
    text.setFont("Courier", 10)
    for line in steps.split("\n"):
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 100, f"Final Answer: {final_answer}")

    c.showPage()
    c.save()
    return filename

# ---------------- INTERACTIVE GRAPHING ----------------
def plot_function_with_roots(expr, sol, x_range=(-10, 10)):
    x = sp.Symbol("x")
    f = sp.lambdify(x, expr, "numpy")

    # Generate data points
    X = np.linspace(x_range[0], x_range[1], 500)
    Y = f(X)

    # Base function plot
    fig = px.line(
        x=X, y=Y,
        labels={"x": "x", "y": "f(x)"},
        title="📉 Function Graph"
    )

    # Highlight roots (if real)
    real_roots = []
    if isinstance(sol, list):
        for r in sol:
            if hasattr(r, "is_real") and r.is_real:
                try:
                    val = float(r)
                    if x_range[0] <= val <= x_range[1]:
                        real_roots.append(val)
                except:
                    pass

    if real_roots:
        fig.add_scatter(
            x=real_roots,
            y=[0]*len(real_roots),
            mode="markers",
            marker=dict(color="red", size=10),
            name="Roots"
        )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Math Assistant Chatbot", page_icon="🧮")

st.title("🧮 Math Assistant Chatbot")
st.write("Type your math problem OR upload a screenshot, and I’ll solve step by step!")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

text_query = st.chat_input("Type your math problem here... (e.g. solve x^2 - 5x + 6 = 0)")
if text_query:
    query = text_query

if query:
    st.write("**Original Input:**", query)
    st.write("**LaTeX Preview:**")
    st.latex(clean_equation(query))

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            raw_response = agent_executor.invoke({"query": query})
            try:
                structured_response = parser.parse(raw_response.get("output"))

                st.markdown(f"**Problem:** {structured_response.problem}")
                st.markdown("**Steps:**")
                for line in structured_response.solution_steps.split("\n"):
                    if line.strip().startswith("$"):
                        st.latex(line.strip("$"))
                    else:
                        st.write(line)

                st.markdown("**Final Answer:**")
                st.latex(structured_response.final_answer)

                # Interactive Plot
                try:
                    x = sp.Symbol("x")
                    cleaned_expr = clean_equation(query.replace("solve", "").strip())
                    expr = sp.sympify(cleaned_expr)

                    # User slider for range
                    x_min, x_max = st.slider("Select x-axis range", -50, 50, (-10, 10))
                    plot_function_with_roots(expr, sp.solve(expr, x), (x_min, x_max))
                except Exception as e:
                    st.warning(f"Could not plot graph: {e}")

                # PDF Export
                pdf_file = save_solution_as_pdf(
                    structured_response.problem,
                    structured_response.solution_steps,
                    structured_response.final_answer
                )
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="📄 Download Solution as PDF",
                        data=f,
                        file_name="solution.pdf",
                        mime="application/pdf"
                    )

                response_text = f"Problem: {structured_response.problem} | Final Answer: {structured_response.final_answer}"
            except Exception:
                response_text = raw_response.get("output", "Sorry, I couldn’t parse the response.")

    st.session_state.messages.append({"role": "assistant", "content": response_text})
