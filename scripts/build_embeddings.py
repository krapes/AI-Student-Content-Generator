from langchain.document_loaders import DataFrameLoader
import pandas as pd
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
global CHROMA_PERSIST_DIRECTORY

_ = load_dotenv(find_dotenv())  # read local .env fil
persist_directory = '../data/chroma'

def open_pickel(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)  # deserialize using load()
    return file
df = open_pickel("../data/CCSS_standards_dataframe.pkl")

def concat_row(row):
    new_dict = {}
    for key in row.keys():
        new_dict[key] = row[key]
    return str(new_dict)

df['concat_text'] = df.apply(concat_row, axis=1)



loader = DataFrameLoader(df, page_content_column="Standard")


embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=loader.load(),
    embedding=embedding,
    persist_directory=persist_directory
)
vectordb.persist()
print(vectordb._collection.count())
ccss_input = 'CCSS.ELA-LITERACY.W.4.9 '
docs = vectordb.similarity_search(ccss_input,k=1)
for doc in docs:
    print(doc.page_content)