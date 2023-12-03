#test-rag.py
#pip install langchainhub gradio

import os
from dotenv import load_dotenv
import json

import gradio as gr

from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, pipeline

#from googletrans import Translator

from optimum.intel.openvino import OVModelForCausalLM


load_dotenv(verbose=True)
model_vendor     = os.environ['MODEL_VENDOR']
model_name       = os.environ['MODEL_NAME']
model_precision  = os.environ['MODEL_PRECISION']
inference_device = os.environ['INFERENCE_DEVICE']
document_dir     = os.environ['DOCUMENT_DIR']
vectorstore_dir  = os.environ['VECTOR_DB_DIR']
num_max_tokens   = int(os.environ['NUM_MAX_TOKENS'])
embeddings_model = os.environ['MODEL_EMBEDDINGS']
ov_config        = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":"./cache"}

### WORKAROUND for "trust_remote_code=True is required error" in HuggingFaceEmbeddings()
from transformers import AutoModel
model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True) 

embeddings = HuggingFaceEmbeddings(
    model_name = embeddings_model,
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)

vectorstore_dir = f'{vectorstore_dir}_300_0'
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

ov_model_path = f'./{model_name}/{model_precision}'
model = OVModelForCausalLM.from_pretrained(
    model_id  = ov_model_path,
    device    = inference_device,
    ov_config = ov_config,
    cache_dir = './cache'
)

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens
)

llm = HuggingFacePipeline(pipeline=pipe)

chain_type =  ['stuff', 'map_reduce', 'map_rerank', 'refine'][0]
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever)

# -------------------------

def run_generation(text_user_en):
    #translator = Translator()
    #query_en = translator.translate(text_user_jp, src='ja', dest='en')
    #query_en = query_en.text
    ans = qa_chain.run(text_user_en)
    #translation_jp = translator.translate(ans, src='en', dest='ja')
    return ans

def reset_textbox():
    print('Clear')
    return '', ''


with gr.Blocks() as demo:
    gr.Markdown('<h1><center>OpenVINO Q&A Chatbot</center></h1>')
    with gr.Row():
        with gr.Column(scale=2):
            #text_user_jp = gr.Textbox(label='Question(JP)', placeholder='Type in your question here.')
            text_user_en = gr.Textbox(label='Question(EN)', placeholder='')
    with gr.Row():
        with gr.Column(scale=2):
            #text_answer_jp = gr.Textbox(label='Answer(JP)', placeholder='Response from Q&A system', interactive=False)
            text_answer_en = gr.Textbox(label='Answer(EN)', placeholder='', interactive=False)
    with gr.Row():
        button_submit = gr.Button(value='Submit',)
        button_clear  = gr.Button(value='Clear')
    examples = [
        ['What is NNCF?'],
        ['Explain OpenVINO briefly.'],
    ]
    gr.Examples(examples=examples, inputs=text_user_en)

    text_user_en.submit(fn=run_generation, inputs=[text_user_en], outputs=[text_answer_en])
    button_submit.click(fn=run_generation, inputs=[text_user_en], outputs=[text_answer_en])
    button_clear.click(fn=reset_textbox, outputs=[text_user_en, text_answer_en])

demo.queue()
demo.launch(share=True, inbrowser=True)
