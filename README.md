# News Research Tool 📈

A powerful web application that lets you extract, analyze, and question content from multiple online articles using advanced language models and embeddings.

## Overview

This News Research Tool enables users to:
- Load multiple URLs containing news articles or other web content
- Process and analyze the content using embeddings and vector search
- Ask natural language questions about the loaded content
- Get AI-generated answers with source citations

## Features

- **Multiple URL Support**: Analyze up to 3 URLs simultaneously
- **Advanced Text Processing**: Uses RecursiveCharacterTextSplitter to break down documents into manageable chunks
- **Semantic Search**: Employs FAISS with HuggingFace embeddings to find relevant information
- **AI-Powered Question Answering**: Leverages Google's Gemini 1.5 Pro model to generate accurate responses
- **Source Citation**: Answers include references to the source documents
- **User-Friendly Interface**: Built with Streamlit for easy interaction

## Prerequisites

Before running this application, you'll need:

- Python 3.7+
- A Google API key for Gemini access
- Required Python packages (see Installation section)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/news-research-tool.git
   cd news-research-tool
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. In the sidebar, enter up to 3 URLs containing the content you want to analyze.

3. Click the "Process URLs" button to extract and analyze the content.

4. Once processing is complete, use the main text input field to ask questions about the content.

5. The application will provide answers based on the analyzed content, with source citations.

## How It Works

1. **Data Loading**: URLs are processed using WebBaseLoader to extract content.
2. **Text Splitting**: The content is split into smaller chunks for more effective processing.
3. **Embeddings Generation**: HuggingFace embeddings are created for each text chunk.
4. **Vector Storage**: Embeddings are stored in a FAISS index for efficient retrieval.
5. **Question Answering**: When a user asks a question, the system:
   - Finds the most relevant text chunks using the FAISS index
   - Passes these chunks to the Gemini model along with the question
   - Returns the model's answer with source citations

## Notes

- The application uses HuggingFace embeddings instead of OpenAI embeddings to reduce costs.
- If the application doesn't load content properly, check if the URLs are accessible and contain readable text.
- Large or complex webpages might take longer to process.

## Requirements

- streamlit
- langchain
- google-generativeai
- faiss-cpu
- sentence-transformers
- python-dotenv
- unstructured
- requests
- bs4

## License

[Add your license information here]

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.