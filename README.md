# Q&A Chatbot for OpenVINO web documentation by OpenVINO

This is an example of an LLM based Q&A chatbot that can refer to external documents using RAG (Retrieval Augmented Genration) technique. The program uses OpenVINO as the inferencing acceleration library.

The program can answer your questions by referring the OpenVINO technical documentation from the OpenVINO official web site.

This program doesn't rely on any cloud services or webAPIs for inferencing. The program downloads all the data, including reference documents and DL models, and **can perform inference offline**. You don't need any cloud services once you prepare the data locally. 

## Programs / Files

|#|Program/File|Description|
|---|---|---|
|1|`llm-model-downloader.py`|Download databrics/dolly-2 and meta-llama/llama2-7b-chat models, and convert them into OpenVINO IR models.|
|2|`openvino-doc-specific-extractor.py`|Convert OpenVINO HTML documents into vector store (DB).<br>Reads HTML documents, extracts text, generates embeddings, and store it into vector store.<br>You need to download an archived (zipped) HTML document from OpenVINO document web site.|
|3|`openvino-rag-server.py`|OpenVINO Q&A demo server|
|4|`openvino-rag-client.py`|OpenVION Q&A demo client|
|5|`.env`|Configurations (no secrets nor credentials ncluded. just a configuration file)|
|6|`requirements.txt`|Python module requirements file|
|7|`huggingface_login.py`|(optional) A Python script to login to HuggingFace hub.|

## How to run

0. Install Python prerequisites

Install steps for Windows.
```sh
python -m venv venv
venv/Scripts/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt

# Install en_core_web_sm, a Spacy pipeline for English
python -m spacy download en_core_web_sm
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
	- '_300_0' means the chunk size is 300 and chunk overlap is 0.
	- You can generate the vector store with different chunk configurations by modifying the last few lines of Python code.
	- You can modify the `.env` file to specify which vector store file to use in the client and server programs. 

3. Download LLM models and convert them into OpenVINO IR models
- `llm-model-downloader.py` will download 'dolly2-3b', 'llama2-7b-chat', and 'Intel/neural-chat-7b-v3-1' models as default.
	- You can specify the LLM model to use by modifying `.env` file.
- You need to have account and access token to download the 'llama2-7b-chat' model. Go to HuggingFace web site and register yourself to get the access token. Also, you need to request the access to the llama2 models at llama2 project page.
- The downloader will generate FP16, INT8 and INT4 models by default. You can use one of them. Please modify `.env` file to specify which model of data type to use.
```sh
python llm-model-downloader.py
```

4. Run the demo
- Run the server
- Note: The '`--host 0.0.0.0`' option is to accept external connection. '`--port xx`' option is also available.
```sh
uvicorn openvino-rag-server:app --host 0.0.0.0
```
- Run the client
- Note: You can change the server URL (or IP address) and port number by editing `.env` file.
```sh
streamlit run openvino-rag-client.py
``` 
Note: You can start the server and client in arbitrary order.

## Examples
![pic1](./resources/screenshot1.png)

## Tested environment
- OS: Windows 11
- OpenVINO: OpenVINO 2023.2.0
