# Q&A Chatbot for OpenVINO web documentation by OpenVINO

This is an example of an LLM based Q&A chatbot that can refer to external documents using RAG (Retrieval Augmented Genration) technique. The program uses OpenVINO as the inferencing acceleration library.

The program can answer your questions by referring the OpenVINO technical documentation from the OpenVINO official web site.

## Programs / Files

|#|Program/File|Description|
|---|---|---|
|1|llm-model-downloader.py|Download databrics/dolly-2 and meta-llama/llama2-7b-chat models, and convert them into OpenVINO IR models.|
|2|openvino-doc-specific-extractor.py|Convert OpenVINO HTML documents into vector store (DB).<br>Reads HTML documents, extracts text, generates embeddings, and store it into vector store.|
|3|openvino-rag-server.py|OpenVINO Q&A demo server|
|4|openvino-rag-client.py|OpenVION Q&A demo client|
|5|.env|Configurations (no secrets nor credentials ncluded. just a configuration file)|
|6|requirements.txt|Python module requirements file|

## How to run

0. Install Python prerequisites
(Win)
```sh
python -m venv venv
venv/Scripts/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
```

1. Downloading OpenVINO Documents
- Go to [OpenVINO web document page](https://docs.openvino.ai/2023.2/get_started.html) and download the archived document file from 'Download Docs' link on the right.
- https://docs.openvino.ai/2023.2/get_started.html
- Extract the contents of downloaded zip file into '`openvino_html_doc`' folder

2. Generate vector store from the OpenVINO documents
- Run '`openvino-doc-specific-extractor.py`'.
- The program will store the document object in a pickle file (`doc_obj.pickle`) and use it if it exists the next time.
```sh
python openvino-doc-specific-extractor.py
```
- `.vectorstore_300_0` directory will be created.

3. Download LLM models and convert them into OpenVINO IR models
- `llm-model-downloader.py` will download 'dolly2-3b' and 'llama2-7b-chat' models.
- You need to have account and access token to download the 'llama2-7b-chat' model. Go to HuggingFace web site and register yourself to get the access token. Also, you need to request the access to the llama2 models at llama2 project page.
- The downloader will generate FP16, INT8 and INT4_ASYM models. You can use one of them.
```sh
python llm-model-downloader.py
```

4. Run the demo
- Run the server
```sh
uvicorn openvino-rag-server:app
```
- Run the client
```sh
streamlit run openvino-rag-client.py
``` 

## Tested environment
- OS: Windows 11
- OpenVINO: OpenVINO 2023.2.0
