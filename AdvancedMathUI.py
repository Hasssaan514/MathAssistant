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
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO
import time

# Load environment variables
load_dotenv()


# ----------------- ENHANCED STYLING -----------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }

    /* Custom header styling */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .custom-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }

    /* Input area styling */
    .input-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }

    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: #764ba2;
        background: rgba(255,255,255,1);
    }

    /* Chat messages styling */
    .stChatMessage {
        margin-bottom: 1rem;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Success/info messages */
    .stSuccess, .stInfo {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Math display area */
    .math-display {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }

    /* Solution steps */
    .solution-step {
        background: rgba(255,255,255,0.8);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)


# ----------------- EQUATION CLEANER -----------------
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


# ----------------- ENHANCED SOLVER TOOL -----------------
@tool
def solve_equation(equation: str) -> str:
    """Solve mathematical equations with detailed step-by-step solutions"""
    try:
        x = sp.symbols("x")
        steps = []
        cleaned = clean_equation(equation)

        if "=" in cleaned:
            expr = sp.sympify(cleaned.split("=")[0]) - sp.sympify(cleaned.split("=")[1])
        else:
            expr = sp.sympify(cleaned)

        sol = sp.solve(expr, x)

        # Enhanced step formatting
        steps.append(f"Original: ${sp.latex(expr)} = 0$")
        steps.append(f"Simplified: ${sp.latex(sp.simplify(expr))} = 0$")

        if sol:
            if len(sol) == 1:
                steps.append(f"Solution: $x = {sp.latex(sol[0])}$")
            else:
                sol_str = ", ".join([f"x = {sp.latex(s)}" for s in sol])
                steps.append(f"Solutions: ${sol_str}$")
        else:
            steps.append("No real solutions found")

        return "\n".join(steps)
    except Exception as e:
        return f"Error solving equation: {str(e)}"


# ----------------- RESPONSE MODEL -----------------
class MathResponse(BaseModel):
    problem: str
    solution_steps: str
    final_answer: str


# ----------------- LLM SETUP -----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=MathResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
     You are an expert Math Tutor Assistant. Solve mathematical problems step-by-step.
     If given an image, extract the math problem first.
     Provide clear, educational explanations like a patient teacher.
     Format your response as: {format_instructions}
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [solve_equation]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------- STREAMLIT APP -----------------
st.set_page_config(
    page_title="Math AI Tutor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# ----------------- HEADER -----------------
st.markdown("""
<div class="custom-header">
    <h1>🧮 AI Math Tutor</h1>
    <p>Your intelligent companion for solving mathematical problems step-by-step</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 📚 Features")
    st.markdown("""
    - 📸 **Image Recognition**: Upload problem screenshots
    - 🔢 **Step-by-Step Solutions**: Detailed explanations
    - 📊 **Graph Plotting**: Visual function representations
    - 🎯 **Multiple Input Methods**: Text or image input
    - 📝 **LaTeX Rendering**: Beautiful mathematical notation
    """)

    st.markdown("---")
    st.markdown("### 🎯 Supported Topics")
    topics = ["Algebra", "Quadratic Equations", "Trigonometry", "Linear Equations", "Polynomials", "Functions"]
    for topic in topics:
        st.markdown(f"• {topic}")

    st.markdown("---")
    st.markdown("### 💡 Example Problems")
    examples = [
        "x² - 5x + 6 = 0",
        "sin(x) = 0.5",
        "2x + 3 = 7",
        "x³ - 8 = 0"
    ]
    for example in examples:
        if st.button(f"Try: {example}", key=f"example_{example}"):
            st.session_state.example_query = example

# ----------------- MAIN CONTENT -----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Input Your Problem")

    # Tab interface for different input methods
    tab1, tab2 = st.tabs(["✏️ Type Problem", "📸 Upload Image"])

    with tab1:
        # Check for example query from sidebar
        default_value = ""
        if hasattr(st.session_state, 'example_query'):
            default_value = st.session_state.example_query
            delattr(st.session_state, 'example_query')

        text_query = st.text_area(
            "Enter your math problem:",
            placeholder="e.g., solve x² - 5x + 6 = 0",
            height=100,
            value=default_value
        )

        col_solve, col_clear = st.columns([1, 1])
        with col_solve:
            solve_text = st.button("🚀 Solve Problem", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a screenshot of your math problem",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="📷 Uploaded Problem", use_column_width=True)

            with st.spinner("🔍 Extracting equation from image..."):
                time.sleep(1)  # Simulate processing time for better UX
                img_base64 = base64.b64encode(
                    BytesIO(uploaded_file.getvalue()).getvalue()
                ).decode("utf-8")

                try:
                    vision_response = llm.invoke([
                        HumanMessage(content=[
                            {"type": "text",
                             "text": "Extract the math equation from this image and return in plain text format."},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
                        ])
                    ])
                    extracted_query = vision_response.content.strip()
                    st.success(f"✅ Extracted: **{extracted_query}**")

                    if st.button("🚀 Solve Extracted Problem", type="primary"):
                        st.session_state.current_query = extracted_query
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error extracting equation: {str(e)}")

with col2:
    st.markdown("### 📊 Quick Stats")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.problems_solved = 0

    # Stats display
    st.metric("Problems Solved", st.session_state.problems_solved)
    st.metric("Chat Messages", len(st.session_state.messages))

    # Progress indicator
    if st.session_state.problems_solved > 0:
        st.progress(min(st.session_state.problems_solved / 10, 1.0))
        st.caption(f"Progress toward 10 problems: {st.session_state.problems_solved}/10")


# ----------------- PROBLEM SOLVING LOGIC -----------------
def solve_math_problem(query):
    """Enhanced problem solving with better UI feedback"""

    # Show equation preview
    st.markdown("### 🔍 Problem Analysis")

    col_orig, col_cleaned = st.columns(2)
    with col_orig:
        st.markdown("**Original Input:**")
        st.code(query, language="text")

    with col_cleaned:
        st.markdown("**Cleaned Equation:**")
        cleaned = clean_equation(query)
        try:
            st.latex(cleaned)
        except:
            st.code(cleaned, language="text")

    # Solving animation
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("🔍 Analyzing equation...")
        elif i < 70:
            status_text.text("⚡ Computing solution...")
        else:
            status_text.text("📝 Formatting results...")
        time.sleep(0.01)

    progress_bar.empty()
    status_text.empty()

    # Get solution
    try:
        raw_response = agent_executor.invoke({"query": query})
        structured_response = parser.parse(raw_response.get("output"))

        # Display results in enhanced format
        st.markdown("### 📋 Solution")

        # Problem statement
        st.markdown(f"""
        <div class="math-display">
            <h4>📝 Problem Statement</h4>
            <p>{structured_response.problem}</p>
        </div>
        """, unsafe_allow_html=True)

        # Solution steps
        st.markdown("### 📚 Step-by-Step Solution")
        steps = structured_response.solution_steps.split("\n")

        for i, step in enumerate(steps, 1):
            if step.strip():
                with st.expander(f"Step {i}", expanded=True):
                    if step.strip().startswith("$") and step.strip().endswith("$"):
                        st.latex(step.strip("$"))
                    else:
                        st.write(step)

        # Final answer highlight
        st.markdown("### 🎯 Final Answer")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center;
                    box-shadow: 0 5px 20px rgba(0,0,0,0.2);">
            <h3 style="color: white; margin: 0;">
                {structured_response.final_answer}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Try to plot the function
        try:
            x = sp.Symbol("x")
            expr = sp.sympify(clean_equation(query.replace("solve", "").strip()))

            st.markdown("### 📈 Function Graph")

            # Create better plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x_vals = np.linspace(-10, 10, 1000)
            y_vals = [float(expr.subs(x, val)) for val in x_vals]

            ax.plot(x_vals, y_vals, linewidth=3, color='#667eea')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('f(x)', fontsize=12)
            ax.set_title(f'Graph of {expr}', fontsize=14, fontweight='bold')

            # Highlight solutions
            solutions = sp.solve(expr, x)
            for sol in solutions:
                if sol.is_real:
                    ax.plot(float(sol), 0, 'ro', markersize=10, label=f'x = {sol}')

            if solutions:
                ax.legend()

            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.info("📊 Graph plotting not available for this equation type")

        # Update stats
        st.session_state.problems_solved += 1

        return f"✅ Solved: {structured_response.problem}"

    except Exception as e:
        st.error(f"❌ Error solving problem: {str(e)}")
        return f"❌ Error: {str(e)}"


# ----------------- MAIN APP LOGIC -----------------
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.problems_solved = 0

# Handle query from different sources
query = None

# From text input
if solve_text and text_query:
    query = text_query

# From extracted image
if hasattr(st.session_state, 'current_query'):
    query = st.session_state.current_query
    delattr(st.session_state, 'current_query')

# Process query
if query:
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Solve the problem
    with st.container():
        result = solve_math_problem(query)
        st.session_state.messages.append({"role": "assistant", "content": result})

# ----------------- CHAT HISTORY -----------------
if st.session_state.messages:
    st.markdown("### 💬 Recent Problems")

    # Show only last 3 conversations for cleaner UI
    recent_messages = st.session_state.messages[-6:]  # Last 3 conversations (user + assistant pairs)

    for i in range(0, len(recent_messages), 2):
        if i + 1 < len(recent_messages):
            user_msg = recent_messages[i]
            assistant_msg = recent_messages[i + 1]

            with st.expander(f"🔢 Problem: {user_msg['content'][:50]}..."):
                st.markdown(f"**Your Question:** {user_msg['content']}")
                st.markdown(f"**Result:** {assistant_msg['content']}")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🤖 Powered by Google Gemini AI • 🔧 Built with Streamlit & LangChain</p>
    <p>💡 Tip: Try uploading handwritten math problems for best results!</p>
</div>
""", unsafe_allow_html=True)