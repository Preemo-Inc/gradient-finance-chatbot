import os
import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
import streamlit as st
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index import ServiceContext
from llama_index.llms import GradientBaseModelLLM
from llama_index.embeddings import GradientEmbedding

absolute_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(absolute_dir, "data")

# Replace this with your MONGO_URI_TEMPLATE
MONGO_URI_TEMPLATE = "mongodb+srv://{mongo_user_name}:{mongo_password}@{mongo_cluster_name}.cqf17q7.mongodb.net/?retryWrites=true&w=majority"
DEFAULT_LLM_MODEL_SLUG = "nous-hermes2"
DEFAULT_EMBEDDING_MODEL_SLUG = "bge-large"

mongo_uri = MONGO_URI_TEMPLATE.format(
    mongo_user_name=st.secrets.mongo_user_name,
    mongo_password=st.secrets.mongo_password,
    mongo_cluster_name=st.secrets.mongo_cluster_name,
)

st.set_page_config(
    page_title="Gradient Finance Demo",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Chat about your company's finances, powered by Gradient 🚀")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about your company's finances!",
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data_to_mongodb_atlas():
    with st.spinner(text="Loading and indexing data..."):
        reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        docs = reader.load_data()
        mongodb_client = pymongo.MongoClient(mongo_uri)
        store = MongoDBAtlasVectorSearch(mongodb_client)
        service_context = ServiceContext.from_defaults(
            llm=GradientBaseModelLLM(
                base_model_slug=DEFAULT_LLM_MODEL_SLUG,
                access_token=st.secrets.gradient_access_token,
                workspace_id=st.secrets.gradient_workspace_id,
            ),
            embedding=GradientEmbedding(
                gradient_model_slug=DEFAULT_EMBEDDING_MODEL_SLUG,
                gradient_access_token=st.secrets.gradient_access_token,
                gradient_workspace_id=st.secrets.gradient_workspace_id,
            ),
        )
        storage_context = StorageContext.from_defaults(vector_store=store)
        index = GPTVectorStoreIndex.from_documents(
            docs, storage_context=storage_context, service_context=service_context
        )
        return index


index = load_data_to_mongodb_atlas()

if "chat_engine" not in st.session_state.keys():  # Initialize chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True
    )

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
