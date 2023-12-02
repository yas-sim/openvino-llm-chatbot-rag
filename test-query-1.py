#test-query-1.py

import os
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv(verbose=True)
document_dir = os.environ['DOCUMENT_DIR']
vectorstore_dir = os.environ['VECTOR_DB_DIR']
embeddings_model = os.environ['MODEL_EMBEDDINGS']

### WORKAROUND for "trust_remote_code=True is required error" in HuggingFaceEmbeddings()
from transformers import AutoModel
model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True) 

print('*** Embedding and storing splitted documents into vector store')
embeddings = HuggingFaceEmbeddings(
    model_name = embeddings_model,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

vectorstore_dir = f'{vectorstore_dir}_300_0'
vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)

while True:
    print('Question:', end='')
    query = input()
    query_embedding = embeddings.embed_query(query)
    documents = vectorstore.similarity_search(query, k=10)
    #documents = vectorstore.similarity_search_by_vector(query_embedding, k=10)

    print(f'Number of documents: {len(documents)}')

    for document in documents:
        print(f'Document contents: {document.page_content}')
        print('-'*60)
    print('\n'*2)
