**Vehicle Specification Extraction using RAG + Local LLM**



This project implements a Retrieval-Augmented Generation (RAG) pipeline to extract structured automotive specifications from workshop manuals using a locally hosted LLM (Mistral-7B).





**Problem Statement**



Automotive workshop manuals contain large volumes of unstructured technical data such as torque values, capacities, tools, and specifications. Extracting structured information manually is time consuming and error prone.



This project builds a local **Retrieval-Augmented Generation (RAG)** pipeline that:

* processes large PDF manuals
* retrieves relevant technical context
* uses a locally hosted LLM (Mistral-7B) for structured extraction
* returns machine-readable JSON specifications





**System Architecture**



Pipeline stages:



1. PDF ingestion using PyMuPDF
2. Semantic sentence-based chunking
3. Dense embeddings using Sentence-Transformers (all-mpnet-base-v2)
4. FAISS vector retrieval
5. Local LLM inference (Mistral-7B-Instruct, 4-bit quantized)
6. Structured JSON extraction



This architecture enables:



* offline inference
* low-latency local execution
* scalable retrieval over large manuals
* deterministic structured outputs





**Model Choices**



Embeddings

Sentence-Transformers (all-mpnet-base-v2)



Reason:

* strong semantic similarity performance
* efficient dense vector retrieval
* good performance on technical text



LLM

Mistral-7B-Instruct (local deployment)



Reason:

* strong instruction following
* efficient 4-bit quantization support
* runs on single GPU
* no external API dependency







**Setup :**

pip install -r requirements.txt



**Run** :

Use notebook: notebook/rag\_pipeline.ipynb or python src/pipeline.py







Single query Example: 

Torque specification for brake caliper bolts



Example Output:\[

                {

                  "component": "Brake caliper bolts",

                  "spec\_type": "Torque",

                  "value": "37",

                  "unit": "Nm"

                }

               ]



\*Batch query supported\*





**Research Contributions**



This implementation explores:



* retrieval quality vs chunk size trade-offs
* local LLM vs API-based inference
* deterministic structured prompting for extraction
* FAISS indexing performance on technical documents















