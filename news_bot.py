# Import standard libraries
import os
import re
import json
import getpass
import logging

# Import third-party libraries for web scraping, API interactions, and data processing
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Import libraries for interacting with OpenAI and other language models
import openai
import llama_index
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)

# Import for creating web interfaces
import gradio as gr

# Import specific utilities for news feed parsing and query processing
from RAG_utils import NewsFeedParser, HybridRetriever, NewsQueryEngine

with open('config.json') as config_file:
    config = json.load(config_file)
    
# Setup logging
logging.basicConfig(level=logging.INFO)
openai.api_key = config['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = openai.api_key


llm = OpenAI(model="gpt-4", temperature=0.1, max_tokens=512)
embed_model = OpenAIEmbedding()


def chatbot(input_text):
    # Create an instance of NewsFeedParser and process query
    news_parser = NewsFeedParser()
    documents = news_parser.process_and_chunk_articles(input_text)

    # Initialize the query engine with the processed documents
    pdf_query_engine = NewsQueryEngine(documents, llm, embed_model)
    query_engine = pdf_query_engine.setup_query_engine()

    # Process the query using the query engine
    response = query_engine.query(input_text)
    return response

# Gradio interface setup
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.components.Textbox(lines=3, label="Enter your text:"),
    outputs=gr.components.Textbox(lines=20, label="Answer:"),
    title="FinWise Explorer"
)

# Launch the Gradio interface
iface.launch(share=True)
