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
import plotly.graph_objects as go

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

def extract_equation_for_plotting(query: str) -> str:
    """Extract and clean equation for plotting"""
    query = re.sub(r'^(solve|graph|plot)\s+', '', query.strip(), flags=re.IGNORECASE)
    if '=' in query:
        parts = query.split('=')
        if len(parts) == 2:
            left, right = parts
            if right.strip() == '0':
                return left.strip()
            else:
                return f"({left.strip()}) - ({right.strip()})"
    return query

# ----------------- INTEGRAL HANDLER -----------------
def handle_integral(equation: str):
    """
    Detect and parse definite integrals like ∫_{a}^{b} f(x) dx
    Convert into sympy.integrate(f(x), (x, a, b))
    """
    try:
        match = re.match(r"∫_\{(.+?)\}\^\{(.+?)\}\s*\((.+)\)\s*([a-zA-Z])", equation)
        if match:
            lower, upper, integrand, var = match.groups()
            var = sp.Symbol(var)

            integrand_clean = clean_equation(integrand)
            expr = sp.sympify(integrand_clean)

            result = sp.integrate(expr, (var, sp.sympify(lower), sp.sympify(upper)))

            steps = []
            steps.append(f"Integral Setup: $\\int_{{{lower}}}^{{{upper}}} {sp.latex(expr)} \\, d{var}$")
            steps.append(f"Result: ${sp.latex(result)}$")

            return "\n".join(steps), result
        return None, None
    except Exception as e:
        return None, f"Error parsing integral: {e}"

# ----------------- TOOL: Sympy Solver -----------------
@tool
def solve_equation(equation: str) -> str:
    """Solve algebraic, trigonometric equations, or definite integrals step by step."""
    try:
        # Handle definite integrals first
        if "∫" in equation:
            steps, result = handle_integral(equation)
            if result is not None:
                return steps
            elif isinstance(result, str):
                return result

        x = sp.symbols("x")
        steps = []
        cleaned = clean_equation(equation)

        if "=" in cleaned:
            parts = cleaned.split("=")
            expr = sp.sympify(parts[0]) - sp.sympify(parts[1])
        else:
            expr = sp.sympify(cleaned)

        sol = sp.solve(expr, x)
        steps.append(f"Equation: ${sp.latex(expr)} = 0$")
        steps.append(f"Simplified form: ${sp.latex(sp.simplify(expr))}$")

        if sol:
            steps.append(f"Solutions: {', '.join([f'${sp.latex(s)}$' for s in sol])}")
        else:
            steps.append("No real solutions found")

        return "\n".join(steps)
    except Exception as e:
        return f"Error solving equation: {str(e)}"

# ----------------- Structured Response -----------------
class MathResponse(BaseModel):
    problem: str
    solution_steps: str
    final_answer: str

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

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
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------- PDF Export -----------------
def save_solution_as_pdf(problem, steps, final_answer, filename="solution.pdf"):
    try:
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
        for line in steps.split("\n")[:10]:
            text.textLine(line)
        c.drawText(text)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, 100, f"Final Answer: {final_answer}")
        c.showPage()
        c.save()
        return filename
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Math Assistant Chatbot", page_icon="🧮")
st.title("🧮 Math Assistant Chatbot")
st.write("Type your math problem OR upload a screenshot, and I'll solve step by step!")

if "messages" not in st.session_state:
    st.session_state.messages = []

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# File upload
uploaded_file = st.file_uploader("📸 Upload a screenshot of your math problem", type=["png", "jpg", "jpeg"])
query = None
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Problem", use_column_width=True)
    st.info("⏳ Reading equation from image...")
    img_base64 = encode_image_to_base64(img)
    try:
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
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Text input
text_query = st.chat_input("Type your math problem here... (e.g. solve x^2 - 5x + 6 = 0)")
if text_query:
    query = text_query

# Main processing
if query:
    st.write("**Original Input:**", query)
    try:
        cleaned = clean_equation(query)
        st.write("**LaTeX Preview:**")
        st.latex(cleaned)
    except Exception as e:
        st.warning(f"Could not render LaTeX preview: {e}")

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            try:
                raw_response = agent_executor.invoke({"query": query})
                try:
                    structured_response = parser.parse(raw_response.get("output", ""))
                    st.markdown(f"**Problem:** {structured_response.problem}")
                    st.markdown("**Steps:**")
                    for line in structured_response.solution_steps.split("\n"):
                        if line.strip():
                            if line.strip().startswith("$") and line.strip().endswith("$"):
                                st.latex(line.strip("$"))
                            else:
                                st.write(line)
                    st.markdown("**Final Answer:**")
                    st.latex(structured_response.final_answer)
                    response_text = f"Problem: {structured_response.problem} | Final Answer: {structured_response.final_answer}"
                except Exception:
                    st.write("**Solution:**")
                    st.write(raw_response.get("output", "Sorry, I couldn't parse the response."))
                    response_text = raw_response.get("output", "No response available")

                # PDF Export
                st.markdown("---")
                if st.button("📄 Generate PDF Report"):
                    try:
                        if 'structured_response' in locals():
                            pdf_file = save_solution_as_pdf(
                                structured_response.problem,
                                structured_response.solution_steps,
                                structured_response.final_answer
                            )
                            if pdf_file:
                                with open(pdf_file, "rb") as f:
                                    st.download_button(
                                        label="⬇️ Download Solution PDF",
                                        data=f,
                                        file_name="math_solution.pdf",
                                        mime="application/pdf"
                                    )
                        else:
                            st.warning("No structured solution available for PDF export")
                    except Exception as e:
                        st.error(f"Error generating PDF: {e}")
            except Exception as e:
                st.error(f"Error processing query: {e}")
                response_text = "Sorry, there was an error processing your request."

    if 'response_text' in locals():
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Show chat history
if st.session_state.messages:
    st.markdown("---")
    st.subheader("💬 Chat History")
    for i, message in enumerate(st.session_state.messages[-6:]):
        with st.expander(f"{'🧑‍💻 User' if message['role']=='user' else '🤖 Assistant'} - Message {len(st.session_state.messages) - 5 + i}"):
            st.write(message["content"])

if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
