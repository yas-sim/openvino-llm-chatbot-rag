#test-rag.py
#pip install langchainhub

import os
from dotenv import load_dotenv
import json

from langchain import hub
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

from transformers import AutoTokenizer, pipeline

from googletrans import Translator

from optimum.intel.openvino import OVModelForCausalLM


load_dotenv(verbose=True)
model_vendor = os.environ['MODEL_VENDOR']
model_name = os.environ['MODEL_NAME']
document_dir = os.environ['DOCUMENT_DIR']
vectorstore_dir = os.environ['VECTOR_DB_DIR']
ov_config = json.loads(os.environ['OV_CONFIG'])
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

vectorstore_dir = f'{vectorstore_dir}_500_100'
vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
#    search_type='similarity_score_threshold', 
#    search_kwargs={
#        'score_threshold' : 0.8, 
#        'k' : 4
#    }
#)
print(f'** Vector store : {vectorstore_dir}')

model_id = f'{model_vendor}/{model_name}'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir='./cache')

ov_model_path = f'./{model_name}/INT8'
model = OVModelForCausalLM.from_pretrained(ov_model_path, device='CPU', ov_config=ov_config)

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=140
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt = hub.pull("rlm/rag-prompt")
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------

translator = Translator()

while True:
    print('\n'+'*'*40)
    print('QUERY : ', end='')
    query_jp = input()

    query_en = translator.translate(query_jp, src='ja', dest='en')
    print(f'TRANSLATION : {query_en.text}\n')

    ans = rag_chain.stream(query_en.text)
    for chunk in ans:
        translation_jp = translator.translate(chunk, src='en', dest='ja')
        print(f'ANSWER_E : {chunk}')
        print(f'ANSWER_J : {translation_jp.text}')

    print('\n'*2)
