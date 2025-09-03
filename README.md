# 🧮 Math Assistant - AI-Powered Mathematical Problem Solver

An intelligent math tutoring system that combines multiple AI agents to solve mathematical problems step-by-step. This project features both research capabilities and advanced mathematical problem-solving with visual equation recognition.

## ✨ LIVE DEMO: [Try the app here](https://mathassistant-mvygxunkngsoih4nburiju.streamlit.app/)
## ✨ Features

### 🔍 Research Assistant (`appinterface.py`)
- Web search integration for research queries
- Structured response formatting
- Interactive Streamlit chat interface
- Source citation and tool tracking

### 🧮 Basic Math Assistant (`MathAssistant.py`)
- Step-by-step equation solving
- LaTeX rendering for mathematical expressions
- Interactive problem-solving interface
- Support for algebra, trigonometry, and geometry

### 📸 Advanced Math Assistant (`WolfChat.py`)
- **Image Recognition**: Upload screenshots of math problems
- **Equation Cleaning**: Automatically formats messy equations
- **Visual Graphs**: Plots function graphs automatically
- **Multi-input Support**: Text or image input
- **Enhanced LaTeX Preview**: See equations before solving

### 🛠️ Command Line Interface (`main.py`)
- Terminal-based research assistant
- Direct API integration testing
- Structured output parsing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API Key (for Gemini)
- Required Python packages (see Installation)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Hasssaan514/MathAssistant.git
cd MathAssistant
```

2. **Install dependencies:**
```bash
pip install streamlit langchain langchain-google-genai langchain-community
pip install sympy matplotlib pillow pydantic python-dotenv
```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Google API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 🎯 Usage

#### Math Assistant (Recommended)
```bash
streamlit run WolfChat.py
```
- Upload math problem screenshots OR type equations
- Get step-by-step solutions with graphs
- LaTeX-rendered mathematical expressions

#### Research Assistant
```bash
streamlit run appinterface.py
```
- Ask research questions
- Get web search results with citations
- Structured research responses

#### Basic Math Solver
```bash
streamlit run MathAssistant.py
```
- Text-based math problem solving
- Clean step-by-step solutions

#### Command Line Testing
```bash
python main.py
```
- Terminal-based research queries
- API testing and debugging

## 📁 Project Structure

```
MathAssistant/
│
├── WolfChat.py          # 🌟 Advanced Math Assistant (with image support)
├── MathAssistant.py     # 🧮 Basic Math Solver
├── appinterface.py      # 🔍 Research Assistant
├── main.py              # 🛠️ Command Line Interface
├── tools.py             # 🔧 Search tools and utilities
├── AIagent.py           # (Empty - placeholder for future features)
├── .env                 # 🔒 Environment variables (not tracked)
├── .env.example         # 📝 Template for environment setup
├── .gitignore           # 🚫 Git ignore rules
└── README.md            # 📖 This file
```

## 🔧 Configuration

### Required Environment Variables

Create a `.env` file with:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### Getting Google API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy and paste into your `.env` file

## 🎨 Features in Detail

### 📸 Image Recognition (WolfChat.py)
- Upload screenshots of handwritten or printed math problems
- Automatic equation extraction using Gemini Vision
- LaTeX preview before solving
- Support for complex mathematical notation

### 🔍 Web Research (appinterface.py)
- DuckDuckGo search integration
- Structured research responses
- Source tracking and citation
- Perfect for research paper generation

### 📊 Mathematical Capabilities
- **Algebra**: Linear/quadratic equations, polynomials
- **Trigonometry**: Sin, cos, tan functions and equations
- **Geometry**: Basic geometric calculations
- **Graphing**: Automatic function plotting
- **Step-by-step**: Detailed solution explanations

## 🌐 Example Usage

### Solving Equations
```
Input: "solve x^2 - 5x + 6 = 0"
Output: Step-by-step solution with graph
```

### Image Problems
1. Upload screenshot of math problem
2. System extracts equation automatically
3. Provides complete solution with visualization

### Research Queries
```
Input: "Research quantum computing applications"
Output: Structured research with sources and citations
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the `.env.example` for proper configuration
- Ensure all dependencies are installed correctly

## 🔮 Future Features

- [ ] Support for calculus problems
- [ ] 3D graph plotting
- [ ] Mathematical proof generation
- [ ] Multi-language equation support
- [ ] Integration with more math libraries

---

**Built with ❤️ using LangChain, Streamlit, and Google Gemini AI**
