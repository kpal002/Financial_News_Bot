import feedparser
from urllib.parse import quote
from newspaper import Article
from urllib.parse import quote
from llama_index import Document
from typing import Any, List, Tuple
from datetime import datetime, timedelta


from llama_index import PromptTemplate
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import get_response_synthesizer
from llama_index.schema import NodeWithScore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms.base import llm_completion_callback
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import VectorIndexRetriever, BaseRetriever, BM25Retriever

class NewsFeedParser:
    def __init__(self):
        """
        Initializes the NewsFeedParser class.
        """
        self.articles_data = []

    def create_search_urls(self, question, base_urls):
        """
        Converts a question into properly formatted search URLs for multiple base URLs.

        Parameters:
        question (str): The query or question to be converted into search URLs.
        base_urls (list): A list of base URLs for the search services.

        Returns:
        list: A list of formatted search URLs, each containing the encoded question.
        """
        # URL Encoding
        encoded_question = quote(question)

        # Constructing the full URLs for each base URL
        search_urls = [base_url + encoded_question for base_url in base_urls]
        
        return search_urls

    def parse_feed(self, rss_url):
        """
        Parses the RSS feed from a given URL and processes each entry.

        Parameters:
        rss_url (str): URL of the RSS feed to be parsed.
        """
        news_feed = feedparser.parse(rss_url)
        content = news_feed.entries

        # Get the current date
        current_date = datetime.now()

        for entry in content:
            # Extract and format the publication date
            newformat = "%a, %d %b %Y %H:%M:%S %Z"
            published_date = datetime.strptime(entry.published, newformat)

            # Check if the article is within the last week
            if current_date - published_date <= timedelta(days=7):
                # Extract the article text
                article_text = self.extract_article_text(entry.link)

                # Only add to the list if article text is successfully extracted
                if article_text:
                    self.articles_data.append({
                        'link': entry.link,
                        'published': published_date.strftime("%Y-%m-%d %H:%M:%S"),
                        'article_text': article_text
                    })

    def extract_article_text(self, url):
        """
        Extracts text from a given article URL.

        Parameters:
        url (str): The URL of the article from which to extract text.

        Returns:
        str: Extracted article text. Returns None if extraction fails.
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def process_query(self, input_query):
        """
        Processes an input query to extract articles related to the query from multiple sources.

        Parameters:
        input_query (str): The query from which to extract information.

        Returns:
        list: A list of articles with their details.
        """
        # Define base URLs for multiple news sources
        base_urls = [
            'https://news.google.com/rss/search?q=',
            'http://www.ft.com/rss/markets?q='
            #'https://www.bloomberg.com/search?query='
        ]

        # Step 1: Create search URLs for each base URL
        search_urls = self.create_search_urls(input_query, base_urls)
        print(search_urls)

        # Step 2: Parse the feed for each search URL
        for url in search_urls:
            self.parse_feed(url)

        # Return the accumulated articles from all feeds
        return self.articles_data
        
    def chunk_text_by_words_with_overlap(self, text, max_words, overlap, metadata):
        """
        Splits the text into chunks of a specified number of words with a specified overlap
        and attaches metadata to each chunk.
        """
        words = text.split()
        chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words - overlap)]

        # Ensure the last chunk doesn't exceed the text length
        if len(chunks) > 1 and len(chunks[-1].split()) < overlap:
            chunks[-2] = ' '.join(chunks[-2:])
            chunks.pop(-1)

        return [{'text': chunk, **metadata} for chunk in chunks]

    def process_and_chunk_articles(self, input_query, max_words=250, overlap=20):
        """
        Processes an input query, fetches related articles, and chunks their text.
        Returns a list of Document objects with attached metadata.
        """
        # Process the query and get articles
        articles = self.process_query(input_query)
        print(len(articles))

        # Chunk each article's text and create Document objects
        documents = []
        metadata_list = []
        for article in articles:
            article_chunks = self.chunk_text_by_words_with_overlap(
                article['article_text'],
                max_words,
                overlap,
                metadata={'link': article['link'], 'published': article['published']}
            )

            for chunk in article_chunks:
                documents.append(Document(text=chunk['text']))
                metadata_list.append({'link': chunk['link'], 'published_date': chunk['published']})

        # Add metadata to each document
        for doc, meta in zip(documents, metadata_list):
            doc.metadata = meta

        return documents
            
            
class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines results from two different retrieval methods:
    vector-based retrieval and BM25 retrieval.

    Attributes:
        vector_retriever: An instance of a retriever that uses vector embeddings for retrieval.
        bm25_retriever: An instance of a retriever that uses BM25 algorithm for retrieval.

    The class inherits from BaseRetriever, indicating that it follows a similar interface.
    """

    def __init__(self, vector_retriever, bm25_retriever):
        """
        Initializes the HybridRetriever with two different types of retrievers.

        Args:
            vector_retriever: The retriever instance which uses vector-based retrieval methods.
            bm25_retriever: The retriever instance which uses BM25 algorithm for retrieval.
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        """
        Performs a retrieval operation by combining results from both the vector and BM25 retrievers.

        Args:
            query: The query string based on which the documents are to be retrieved.
            **kwargs: Additional keyword arguments that might be required for retrieval.

        Returns:
            all_nodes: A list of nodes (documents) retrieved by combining results from both retrievers.
                       This ensures a diverse set of results leveraging the strengths of both retrieval methods.
        """
        # Retrieve nodes using BM25 retriever
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)

        # Retrieve nodes using vector retriever
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # Combine the two lists of nodes, ensuring no duplicates
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            # Check if node is already added; if not, add it to the list
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)

        return all_nodes
        

class NewsQueryEngine:
    """
    A class to handle the process of setting up a query engine and performing queries on PDF documents.

    This class encapsulates the functionality of creating prompt templates, embedding models, service contexts,
    indexes, hybrid retrievers, response synthesizers, and executing queries on the set up engine.

    Attributes:
        documents (List): A list of documents to be indexed.
        llm (Language Model): The language model to be used for embeddings and queries.
        qa_prompt_tmpl (str): Template for creating query prompts.
        queries (List[str]): List of queries to be executed.

    Methods:
        setup_query_engine(): Sets up the query engine with all necessary components.
        execute_queries(): Executes the predefined queries and prints the results.
    """

    def __init__(self, documents: List[Any], llm: Any, embed_model: Any):
        self.documents = documents
        self.llm = llm
        self.embed_model = embed_model
        self.qa_prompt_tmpl = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "As an experienced financial analyst and researcher, you are tasked with helping fellow analysts in research using the latest financial news.\n "
            "Your answer will be based on the snippets of latest news provided as context information for each query.\n "
            "For each query, provide a concise answer derived from the information provided in the form of news.\n"
            "Try to not assume any critical information that might impact the answer. \n "
            "Note any major issues in the paper's results and analysis.\n"
            "If a query cannot be answered due to lack of information in the context, state this explicitly.\n"

            "Query: {query_str}\n"
            "Answer:"

        )

    def setup_query_engine(self) -> Any:
        """
        Sets up the query engine by initializing and configuring the embedding model, service context, index,
        hybrid retriever (combining vector and BM25 retrievers), and the response synthesizer. Returns the configured query engine.
        """
        # Initialize the service context with the language model and embedding model
        service_context = ServiceContext.from_defaults(llm=self.llm, embed_model=self.embed_model)

        # Create an index from documents
        index = VectorStoreIndex.from_documents(documents=self.documents, service_context=service_context)
        nodes = service_context.node_parser.get_nodes_from_documents(self.documents)

        # Set up vector and BM25 retrievers
        vector_retriever = index.as_retriever(similarity_top_k=5)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

        # Configure the response synthesizer with the prompt template
        qa_prompt = PromptTemplate(self.qa_prompt_tmpl)
        response_synthesizer = get_response_synthesizer(
            service_context=service_context,
            text_qa_template=qa_prompt,
            response_mode="tree_summarize",
        )

        # Assemble the query engine with a reranker and the synthesizer
        reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[reranker],
            response_synthesizer=response_synthesizer,
        )
        return query_engine
