# Document Analysis LLM Application

This Streamlit application uses LangChain and HuggingFace models to analyze documents and answer questions about their content.

## Features

- Upload and process PDF and CSV documents
- Ask questions about document content
- Get AI-generated answers with confidence scores
- View conversation history and analytics
- Interactive visualizations

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application locally:
```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and main file (app.py)
6. Click "Deploy"

## Project Structure

- `app.py`: Main Streamlit application
- `pages/`: Directory containing different pages of the application
  - `llm.py`: LLM document analysis implementation
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore file
- `.env`: Environment variables (not tracked by git)

## Usage

1. Open the application in your browser
2. Upload a PDF or CSV document
3. Wait for the document to be processed
4. Ask questions about the document content
5. View responses with confidence scores and source references

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies 