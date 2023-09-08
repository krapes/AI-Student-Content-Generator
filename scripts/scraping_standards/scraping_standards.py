# fixes a bug with asyncio and jupyter
import nest_asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup
import pickle
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
import re
import ast
import pandas as pd

nest_asyncio.apply()
scrap = False

_ = load_dotenv(find_dotenv())  # read local .env file
llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

def transform_text(text):
    # Regex pattern to capture the standard and its description
    pattern = r'(CCSS\.(ELA-Literacy|Math)\.(\D+)\.(K|[1-8]|9-10|11-12)\.(\d+))(.*)'

    matches = re.findall(pattern, text)

    result = []
    for match in matches:
        standard, subject, skill, grade, numeration, description = match
        result.append({'Standard': standard,
                       'Numeration': numeration,
                       'Description': description,
                       'Math/ELA': subject,
                       'Skill_abbr': skill,
                       'Grade': grade,
                       'Website': f' https://thecorestandards.org/{subject}/{skill}/{grade}/{numeration}'})

    return result
def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    """# Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")
    sidebar = content.find_all('sidebar')

    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements + sidebar:
        element.decompose()"""
    result = []
    standards = content.find_all(class_='standard')
    if len(standards) > 0:
        article_header = content.find(class_="article-header").get_text()
        pattern = r".*»\s*(.+?)\s*(?=»|$)"
        try:
            article_header = re.findall(pattern, article_header)[0]
        except:
            article_header = ""

        for standard in standards:
            print(standard.get_text())
            standard = transform_text(standard.get_text())[0]
            standard['Skill'] = article_header
            result.append(standard)


    return str(result)


def scrap_site():
    sitemap_loader = SitemapLoader(web_path="https://www.thecorestandards.org/sitemap.xml",
                                   filter_urls=[r"https://thecorestandards.org/ELA-Literacy/RL/K.",
                                                r"https://www.thecorestandards.org/Math/Content/1/MD/."],
                                   parsing_function=remove_nav_and_header_elements)
    return sitemap_loader


def save_pickle(filename, file):
    with open(filename, 'wb') as f:  # open a text file
        pickle.dump(file, f)  # serialize the list
        f.close()


def open_pickel(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)  # deserialize using load()
    return file


if scrap:
    sitemap_loader = scrap_site()
    docs = sitemap_loader.load()
else:
    try:
        docs = open_pickel('raw_docs.pkl')
    except:
        sitemap_loader = scrap_site()
        docs = sitemap_loader.load()

import re





save_pickle('raw_docs.pkl', docs)
with open("scrape_standards_prompt_template.txt") as file:
    scrap_standards_prompt_template = file.read()
with open("ela_literacy_rl_raw_scrape") as file:
    input_example = file.read()
with open("output_example") as file:
    output_example = file.read()

results = []
for doc in docs:
    content = ast.literal_eval(doc.page_content)
    if len(content) > 0:
        results = results + content


df = pd.DataFrame(results)
save_pickle('../../data/CCSS_standards_dataframe.pkl', df)
print(f"results: {results}")
