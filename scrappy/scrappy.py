from vertexai.preview.generative_models import GenerativeModel
from IPython.display import display, Markdown
from dotenv import load_dotenv
import streamlit as st
import logging
from typing import List, Dict
import re
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents.base import Document
from pydantic import BaseModel, ValidationError
import json

#run: gcloud auth application-default login

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# DELETE THIS. I ended up not using this
# source: https://hasanaboulhasan.medium.com/how-to-get-consistent-json-from-google-gemini-with-practical-example-48612ed1ab40

# class DocumentSummary(BaseModel):
#     bullets: List[str]

# def extract_json(text_response):
#     # This pattern matches a string that starts with '{' and ends with '}'
#     pattern = r'\{[^{}]*\}'
#     matches = re.finditer(pattern, text_response)
#     json_objects = []
#     for match in matches:
#         json_str = match.group(0)
#         try:
#             # Validate if the extracted string is valid JSON
#             json_obj = json.loads(json_str)
#             json_objects.append(json_obj)
#         except json.JSONDecodeError:
#             # Extend the search for nested structures
#             extended_json_str = extend_search(text_response, match.span())
#             try:
#                 json_obj = json.loads(extended_json_str)
#                 json_objects.append(json_obj)
#             except json.JSONDecodeError:
#                 # Handle cases where the extraction is not valid JSON
#                 continue
#     if json_objects:
#         return json_objects
#     else:
#         return None  # Or handle this case as you prefer


# def extend_search(text, span):
#     # Extend the search to try to capture nested structures
#     start, end = span
#     nest_count = 0
#     for i in range(start, len(text)):
#         if text[i] == '{':
#             nest_count += 1
#         elif text[i] == '}':
#             nest_count -= 1
#             if nest_count == 0:
#                 return text[start:i+1]
#     return text[start:end]


# def validate_json_with_model(model_class, json_data):
#     """
#     Validates JSON data against a specified Pydantic model.
#     Args:
#         model_class (BaseModel): The Pydantic model class to validate against.
#         json_data (dict or list): JSON data to validate. Can be a dict for a single JSON object, 
#                                   or a list for multiple JSON objects.
#     Returns:
#         list: A list of validated JSON objects that match the Pydantic model.
#         list: A list of errors for JSON objects that do not match the model.
#     """
#     validated_data = []
#     validation_errors = []
#     if isinstance(json_data, list):
#         for item in json_data:
#             try:
#                 model_instance = model_class(**item)
#                 validated_data.append(model_instance.dict())
#             except ValidationError as e:
#                 validation_errors.append({"error": str(e), "data": item})
#     elif isinstance(json_data, dict):
#         try:
#             model_instance = model_class(**json_data)
#             validated_data.append(model_instance.dict())
#         except ValidationError as e:
#             validation_errors.append({"error": str(e), "data": json_data})
#     else:
#         raise ValueError("Invalid JSON data type. Expected dict or list.")
#     return validated_data, validation_errors


# def json_to_pydantic(model_class, json_data):
#     try:
#         model_instance = model_class(**json_data)
#         return model_instance
#     except ValidationError as e:
#         print("Validation error:", e)
#         return None


def prompt_for_queries(question:str, nb_queries:int=3) -> str:
    # given a question, return prompt for google queries
    prompt = f'''You are an assistant tasked with improving Google search 
results. Generate {nb_queries} Google search queries that are similar to 
this question: {question}. The output should be a numbered list of questions and each 
should have a question mark at the end. Do not include anything else in your answer, 
do not repeat the question.'''
    return prompt

def parse_queries(text: str) -> List[str]:
    # parse the google queries, remove the number prefix if any
    lines = re.findall(r"\d+\..*?(?:\n|$)", text)
    return [re.sub(r'\d+\.', '', l).strip() for l in lines]

def search_queries(llm:GenerativeModel, question:str, nb_queries:int=3) -> List[str]:
    # given a question, generate prompt to get google queries and send it to LLM
    logger.info(f'Question: {question}')

    prompt = prompt_for_queries(question, nb_queries)
    logger.info(f'Prompt: {prompt}')

    model_response = llm.generate_content(prompt)
    logger.info(f'Answer: \n{model_response.text}')

    questions = parse_queries(model_response.text)

    for idx, q in enumerate(questions):
        logger.info(f'{idx}. {q}')

    return questions
    
def search_results(search:GoogleSearchAPIWrapper, queries:List[str], nb_results:int=3) -> List[Dict]:
    # given list of google queries, perform search and retrieve search results
    search_results = {}

    for q in queries:
        results = search.results(q, nb_results)
        for r in results:
            search_results[r['link']] = {
                'title': r['title'],
                'snippet': r['snippet']
            }

    return search_results

def extract_docs(search_results:List[Dict]) -> List[Document]:
    # given search result links, parse the corresponding web pages
    loader = AsyncHtmlLoader([link for link in search_results.keys()], ignore_load_errors=True)
    html2text = Html2TextTransformer()
    docs_html = loader.load()
    docs = list(html2text.transform_documents(docs_html))
    return docs


def is_relevant(llm:GenerativeModel, question:str, document:Document) -> str:
    # given a question and a document, ask LLM whether the document is relevant to the question
    # note: the document can be irrelevant if it wasnt parsed correctly
    prompt = f'''You are an assistant tasked with deciding whether a document is relevant
to the following question: {question}.
Here is the document:\n

<Document>
{document.page_content[:32766]}
</Document>

Is the document relevant to the question at hand?
Limit your answer to yes or no. 
Do not add any explanation, do not repeat the question.
Answer either yes or no, nothing more.
'''
    model_response = llm.generate_content(prompt)
    yes_or_no = re.findall('yes|no', model_response.text, re.IGNORECASE)
    print(yes_or_no)
    if len(yes_or_no) > 0:
        if yes_or_no[0].lower().strip() == 'yes':
            return True
    return False

def summarize_document(llm:GenerativeModel, question:str, document:Document) -> str:
    # summarize a document with respect to a given question
    prompt = f'''You are an assistant tasked with summarizing a document as it pertains
    to the following question: {question}.
    You are given the source URL and the document content.
    
    <Url>
    {document.metadata['source']}
    </Url>

    <Document>
    {document.page_content[:32766]}
    </Document>

    Summarize the document using bullet points, in a way that will allow another analyst to 
    thoroughly answer the question using only your summary.

    Please provide a response in the follwing format:
    
    Source: {document.metadata['source']}

    Title: provide a title here.
    
    Summary: 
    - bullet point 1
    - bullet point 2
    - bullet point 3
    - etc.
    '''
    model_response = llm.generate_content(prompt)
    return model_response.text

def answer_question(llm:GenerativeModel, question:str, summaries:List[str], output_template:str) -> str:
    # answer original question given a list of summarized documents and an output template
    # output template should in the following format:
    # field1: description of field1
    # field2: description of field2
    # etc.
    # note: the description of a field can be a "list of [things]"
    prompt = f'''You are an assistant tasked with answering the following question: {question}.
    You are given the following reference summarized documents:

    <Summarized Documents>
    '''

    for idx, s in enumerate(summaries):
        prompt += f'[Summarized Document {idx+1}]\n\n'
        prompt += s
    
    prompt += '''</Summarized Documents>
    You MUST source your answer only from the documents above.

    Your answer should be in JSON format, as follows:
    
    {
        results: [
    '''

    for i in range(3):
        prompt += '{\n'
        prompt += output_template     
        prompt += '\n}'
        if i < 2:
            prompt += ',\n'

    prompt += ']'
    prompt += '\n}'

    prompt += '\nPlease limit your answer to the JSON format described above. Do not include any prefix or suffix words outside of the JSON object.'
    prompt += '\nYou are not limited by the number of characters within the JSON. Your answer should be as thorough and complete as possible.'
    prompt += '\nBut make sure your output is a properly formatted JSON object.'
    prompt += '\nVery important: if there any quotes in the text, use single quotes (not double quotes).'

    logger.info('--- Final prompt ---')
    logger.info(prompt)
    
    model_response = llm.generate_content(prompt)
    return model_response.text

#------------- Streamlit app

nb_queries = st.sidebar.number_input('Number of Google queries:', min_value=1, value=3)
nb_results = st.sidebar.number_input('Number of results per query:', 
                                    min_value=1, 
                                    value=4)

output_template = st.text_area(
    'Provide an output template', 
    value='''name: name of the provider,
url: website of the provider,
summary: summary of the value proposition of this provider,
features: list (array) of features supported by this provider's API
''', 
    height=128, 
    max_chars=None)

question = st.text_input('What topic do you want to research:', 
                        value='Business payment API providers in Canada')

run_bt = st.button('Run')
    
if 'llm' not in st.session_state:
    st.session_state['llm'] = GenerativeModel("gemini-1.0-pro")

if 'search' not in st.session_state:
    st.session_state['search'] = GoogleSearchAPIWrapper(k=nb_results)

llm = st.session_state['llm']
search = st.session_state['search']

if run_bt:
    queries = search_queries(llm, question, nb_queries)

    with st.expander('Queries'):
        for q in queries:
            st.write(q)

    results = search_results(search, queries, nb_results)

    st.write(f'Compiling {len(results)} unique search results...')

    with st.expander('Sources'):
        for link, r in results.items():
            st.write(link)
            st.caption(r['snippet'])
            st.divider()

    st.write('Scraping web pages...')

    docs = extract_docs(results)

    st.write('Summarizing documents...')

    summaries = []

    e = st.empty()

    for i,d in enumerate(docs):
        if is_relevant(llm, question, d):
            s = summarize_document(llm, question, d) 
            summaries.append(s)
            e.caption(f'Unique docs summarized: {i+1}/{len(docs)}')

    answer = answer_question(llm, question, summaries, output_template)
    answer = re.sub('json', '', answer, flags=re.IGNORECASE)

    print(answer)

    with st.expander('Text result'):
        st.markdown(answer)

    with st.expander('JSON result'):
        answer_json = re.sub('`', '', answer)
        st.json(answer_json)





