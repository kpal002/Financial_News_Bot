# Financial_News_Bot
A chatbot using GPT-4 to to help financial analysts leverage LLMs to speed up their research using the latest news.

**Live Application**: A direct link to the hosted application on Hugging Face Spaces is provided for easy access. [Hugging Face Spaces](https://huggingface.co/spaces/kpal002/FinWise_Explorer)

## Data Sources
The information base for each query in FinWise Explorer is created by aggregating data from various reputable financial news RSS feeds. The application uses `feedparser`, a robust Python library, to parse and extract data from the following two feeds:

- Google News (Financial Markets)
- Financial Times (Markets)

## Features
- **Query Processing**: Enter a question related to financial markets, stocks, global economy, and more, and receive detailed, relevant information.
- **News Aggregation**: Aggregates and analyzes data from multiple financial news sources.
- **Data Chunking**: Efficiently processes large articles by breaking them into manageable chunks with overlapping context for better analysis.
- **User-Friendly Interface**: Simple and intuitive interface powered by Gradio, allowing for easy interaction.

## Installation
To set up FinWise Explorer, you need to have Python >= 3.10 installed on your system. Clone this repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/finwise-explorer.git
cd finwise-explorer
pip install -r requirements.txt
