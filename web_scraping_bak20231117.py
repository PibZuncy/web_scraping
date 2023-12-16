from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://www.wsj.com"])
#loader = AsyncChromiumLoader(["https://community.tp-link.com/en/home/forum/topic/189382"])
#loader = AsyncChromiumLoader(["https://github.com/JurajNyiri/pytapo"])

html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["span"])

# Result
print(docs_transformed[0].page_content[0:500])


from langchain.document_loaders import AsyncHtmlLoader
urls = ["https://www.espn.com","https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

print(docs)

from langchain.document_loaders import AsyncHtmlLoader
urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

print(docs)

from langchain.document_transformers import Html2TextTransformer
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
docs_transformed[0].page_content[0:500]

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

from langchain.chains import create_extraction_chain

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter

def scrape_with_playwright(urls, schema):
    
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs,tags_to_extract=["span"])
    print("Extracting content with LLM")
    
    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, 
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    
    # Process the first split 
    extracted_content = extract(
        schema=schema, content=splits[0].page_content
    )
    pprint.pprint(extracted_content)
    return extracted_content

urls = ["https://www.wsj.com"]
extracted_content = scrape_with_playwright(urls, schema=schema)