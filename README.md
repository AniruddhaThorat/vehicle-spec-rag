**Vehicle Specification Extraction using RAG + Local LLM**



This project implements a Retrieval-Augmented Generation (RAG) pipeline to extract structured automotive specifications from workshop manuals using a locally hosted LLM (Mistral-7B).



**Features :**

* PDF ingestion using PyMuPDF
* Semantic chunking using NLTK
* Dense embeddings via Sentence-Transformers (MPNet)
* FAISS vector search retrieval
* Local LLM inference using Mistral-7B-Instruct
* Structured JSON extraction of vehicle specifications



**Pipeline :**

* Load workshop manual PDF
* Chunk text semantically
* Generate embeddings
* Build FAISS index
* Retrieve relevant context
* Run local LLM to extract specs
* Output structured JSON



Example Query : Torque specification for brake caliper bolts

Example Output:\[

&nbsp;               {

&nbsp;                 "component": "Brake caliper bolts",

&nbsp;                 "spec\_type": "Torque",

&nbsp;                 "value": "37",

&nbsp;                 "unit": "Nm"

&nbsp;               }

&nbsp;              ]



**Setup :**

pip install -r requirements.txt



**Run** :

Use notebook: notebook/rag\_pipeline.ipynb or python src/pipeline.py



**Model :**

* LLM: Mistral-7B-Instruct (local)
* Embeddings: sentence-transformers/all-mpnet-base-v2
* Vector DB: FAISS



**Notes**

* Fully local pipeline (no API dependency)
* Works offline once model downloaded
* Designed for production-scale spec extraction
