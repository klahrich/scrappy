# Community lib
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain_google_community import GCSDirectoryLoader
# from langchain_community.vectorstores import Chroma

# Chains
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.chains import LLMChain
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Using Vertex AI
import os
import vertexai
import logging
from google.cloud import aiplatform
# from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain_core.documents import Document
from src.rag.langchain import prompt_template

# SUPABASE as Vector Store Database
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from dotenv import load_dotenv

# Google firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud import firestore
from langchain_google_firestore import FirestoreVectorStore


# Load .env variables
load_dotenv()
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION_ID = os.environ.get("LOCATION_ID")
TIMEOUT = os.environ.get("TIMEOUT", 300)
LOGGING_MODE = os.environ.get("LOGGING_MODE", "INFO")
FIRESTORE_DB = os.environ.get("FIRESTORE_DB", "test")
FIRESTORE_CLIENT = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DB)
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemini-1.5-pro-001")


# File ingestions
def load_documents(
    project_id, gcs_bucket, prefix, text_content, chunk_size=1000, chunk_overlap=0
):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        if gcs_bucket:
            loader = GCSDirectoryLoader(
                project_name=project_id,
                bucket=gcs_bucket,
                prefix=prefix,
            )
            documents = loader.load()
        else:
            documents = (
                Document(
                    page_content=text_content,
                    metadata={"source": prefix},
                ),
            )

        res_docs = text_splitter.split_documents(documents)
        for doc in res_docs:
            doc.metadata["session_id"] = prefix  # Add chat session_id to metadata

        logging.info(f"Loaded # of documents = {len(res_docs)}")

        return res_docs
    except Exception as load_documents_err:
        raise load_documents_err


# def combine_documents_chain(llm):
#     try:
#         document_prompt = prompt_template.document_prompt_template
#         document_variable_name = "context"
#         prompt = PROMPT_SELECTOR.get_prompt(llm)
#         llm_chain = LLMChain(llm=llm, prompt=prompt)
#         combine_docs_chain = StuffDocumentsChain(
#             llm_chain=llm_chain,
#             document_prompt=document_prompt,
#             document_variable_name=document_variable_name,
#         )
#         return combine_docs_chain
#     except Exception as combine_documents_chain_err:
#         raise combine_documents_chain_err


def get_retriever(
    documents,
    embedding,
    session,
    retriver_db,
    search_topk=15,
    supabase_url=None,
    supabase_key=None,
    supabase_chunksize=500,
):
    if retriver_db != "SUPABASE":
        ## Joseph 20240506: Change default vector storage to firestore
        # try:
        #     contracts_vector_db = Chroma.from_documents(collection_name="documents", documents=documents, embedding=embedding)
        # except:
        #     contracts_vector_db = Chroma(collection_name="documents", embedding_function=embedding)
        try:
            ## TODO: Need create index before deployment:
            # gcloud alpha firestore indexes composite create \
            # --project=atbv-sb-datascience \
            # --collection-group=ai_studio_documents \
            # --query-scope=COLLECTION \
            # --field-config field-path=embedding,vector-config='{"dimension":"768", "flat": "{}"}' \
            # --database=test
            contracts_vector_db = FirestoreVectorStore.from_documents(
                client=FIRESTORE_CLIENT,
                collection="ai_studio_documents",
                documents=documents,
                embedding=embedding,
                filters=FieldFilter("metadata.session_id", "==", f"{session}")
            )
        except:
            contracts_vector_db = FirestoreVectorStore(
                client=FIRESTORE_CLIENT,
                collection="ai_studio_documents", 
                embedding=embedding,
                filters=FieldFilter("metadata.session_id", "==", f"{session}")
            )
    else:
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            contracts_vector_db = SupabaseVectorStore.from_documents(
                documents,
                embedding,
                client=supabase,
                table_name="documents",
                query_name="match_documents",
                chunk_size=supabase_chunksize,
            )
        except:
            contracts_vector_db = SupabaseVectorStore(
                client=supabase,
                embedding=embedding,
                table_name="documents",
                query_name="match_documents",
            )
        

    return contracts_vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": search_topk,
            "filter": {"session_id": session},
        },  # Filter search result by session id
    )


def embed_retriver(
    documents,
    retriver_db,
    session,
    supabase_url=None,
    supabase_key=None,
    embedding_model="textembedding-gecko@latest",  # The API accepts a maximum of 3,072 input tokens and outputs 768-dimensional vector embeddings. Use the following parameters for the text embeddings model textembedding-gecko,
    search_topk=15,
):
    try:
        embedding = VertexAIEmbeddings(model_name=embedding_model)
        # contracts_vector_db = Chroma.from_documents(documents, embedding)
        # retriever = contracts_vector_db.as_retriever(
        #     search_type="similarity", search_kwargs={"k": search_topk}
        # )
        retriever = get_retriever(
            documents,
            embedding,
            session,
            retriver_db,
            search_topk,
            supabase_url,
            supabase_key,
        )
        return retriever
    except Exception as combine_documents_chain_err:
        raise combine_documents_chain_err
    
    
def invoke_retriver(llm, retriever, question):
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_template.contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template.qa_system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    ##Joseph 20240521: Chat history not being populated in this version to minimize hallucination.
    chat_history = []
    
    res_answer = rag_chain.invoke({"input": question, "chat_history": chat_history})
    
    return res_answer


def get_vertexai_llm(project_id, location, max_output_tokens, temperature, top_p, top_k, default_model=DEFAULT_MODEL):
    vertexai.init(project=project_id, location=location)

    return VertexAI(
            model_name=default_model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            verbose=True,
        )


async def main_rag_handler(
    project_id,
    location,
    gcs_bucket,
    prefix,  # chat sessionid
    question,
    history,
    default_model="text-bison",
    max_output_tokens=2048,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    retriver_db="FIRESTORE",
    supabase_url=None,
    supabase_key=None,
    text_content="",
    slient_mode=False,
):
    try:
        vertexai.init(project=project_id, location=location)

        llm = VertexAI(
            model_name=default_model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            verbose=True,
        )

        # question_generator_chain = LLMChain(
        #     llm=llm, prompt=prompt_template.prompt_template
        # )
        # combine_docs_chain = combine_documents_chain(llm=llm)

        documents = load_documents(
            project_id=project_id,
            gcs_bucket=gcs_bucket,
            prefix=prefix,
            text_content=text_content,
        )

        retriever = embed_retriver(
            documents, retriver_db, prefix, supabase_url, supabase_key
        )

        if slient_mode:
            return True
        else:
            res_answer = invoke_retriver(llm, retriever, question)

            return res_answer

    except Exception as main_rag_handler_err:
        raise main_rag_handler_err
