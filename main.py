import os
import streamlit as st
import pickle
import time
import google.generativeai as genai
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use alternative loaders
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("News Research Tool 📈")
st.sidebar.title("Load URLs to analyze")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:  # Only add non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
#llm = OpenAI(temperature=0.9, max_tokens=500)

#Use gemini instead of OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model= "gemini-1.5-pro" , #"gemini-pro",
                            google_api_key=os.environ.get("GOOGLE_API_KEY"),
                            temperature=0.5,
                            max_tokens=1000)

if process_url_clicked:
    # Check if URLs list is empty
    if not urls:
        st.error("Please add at least one URL before processing.")
    else:
        try:
            # Load data with selected loader
            main_placeholder.text("Data Loading...Started...✅✅✅")

            # Process each URL individually and combine results
            all_docs = []
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    if docs:
                        all_docs.extend(docs)
                        st.sidebar.write(f"Successfully loaded {url}")
                    else:
                        st.sidebar.warning(f"No content found from {url}")
                except Exception as e:
                    st.sidebar.error(f"Error loading {url}: {str(e)}")
                data = all_docs

            # Check if data was loaded successfully
            if not data:
                st.error("No data was loaded from the URLs. Please check if the URLs are valid.")
            else:
                # Debug: Show document count and first document preview
                st.sidebar.write(f"Loaded {len(data)} documents")
                for i, doc in enumerate(data):
                    content_preview = doc.page_content[:150] + "..." if len(
                        doc.page_content) > 150 else doc.page_content
                    st.sidebar.write(f"Document {i} preview: {content_preview}")
                    st.sidebar.write(f"Document {i} length: {len(doc.page_content)} characters")

                # Text splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )

                main_placeholder.text("Text Splitter...Started...✅✅✅")

                try:
                    docs = text_splitter.split_documents(data)

                    # Debug: Show document chunks
                    st.sidebar.write(f"Created {len(docs)} text chunks")

                    # Check if docs is empty after splitting
                    if not docs:
                        st.error(
                            "No text chunks were created. The content might be too short or in an unsupported format.")
                    else:
                        try:
                            # create embeddings and save it to FAISS index
                            #embeddings = OpenAIEmbeddings()
                            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                            main_placeholder.text("Embedding Vector Started Building...✅✅✅")

                            vectorstore_openai = FAISS.from_documents(docs, embeddings)

                            # Save the FAISS index to a pickle file
                            with open(file_path, "wb") as f:
                                pickle.dump(vectorstore_openai, f)

                            main_placeholder.success("Embeddings created and saved successfully! ✅")
                        except Exception as e:
                            st.error(f"Error creating embeddings: {str(e)}")
                            st.info("Check your OpenAI API key and make sure it has enough credits.")
                except Exception as e:
                    st.error(f"Error during text splitting: {str(e)}")
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")



# Create a custom prompt template
template = """
You are a helpful assistant that provides accurate information based on the given context.

Sources:
{summaries}

Question: {question}

Instructions:
- Answer only based on the provided context
- If you don't know the answer based on the context, say "I don't have enough information"
- Cite the sources used in your answer
- Format your response in a clear, concise manner

Answer:
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["summaries", "question"]
)


query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

                # First create the base QA chain with your custom prompt
                qa_chain = load_qa_with_sources_chain(
                    llm,
                    chain_type="stuff",  # This combines all docs into one prompt
                    prompt=PROMPT
                )

                chain = RetrievalQAWithSourcesChain(
                    combine_documents_chain=qa_chain,
                    retriever=vectorstore.as_retriever()
                )

                result = chain({"question": query}, return_only_outputs=True)
                # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                st.header("Answer")
                st.write(result["answer"])

                # Display sources, if available
                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        st.write(source)
        except Exception as e:
            st.error(f"Error retrieving answer: {str(e)}")
    else:
        st.info("Please process some URLs before asking questions.")