import os

import pypdf
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Semantic:
    
    def __init__(self) -> None:
        # load env variables
        self._tracing = os.environ["LANGSMITH_TRACING"] = "true"
        self._langchain_api_key = os.environ["LANGSMITH_API_KEY"]
        self._open_api_key = os.environ["OPENAI_API_KEY"]
        #self._pinecone_api_key = os.environ["PINECONE_API_KEY"]
        self._vector_store = InMemoryVectorStore(OpenAIEmbeddings())
        self._retriever = self._create_retriever()      
        
    
    def vector_documents(self):
        
        # create docuemts 
        documents = [
            Document(
                page_content="cats are great companion, known for beeing goofy",
                metadata={"source": "mammal-pets-doc"}
            ),
            Document(
                page_content="rabbits have their own personality",
                metadata={"source": "mammal-pets-doc" }
            )
        ]
        
        # generate embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        vector_1 = embeddings.embed_query(documents[0].page_content)
        vector_2 = embeddings.embed_query(documents[1].page_content)
        
        assert len(vector_1) == len(vector_2)
        print(f"Generated vectors of length {len(vector_1)}\n")
        print(vector_1[:10])
        
        # vector store
        self._vector_store = InMemoryVectorStore(embeddings)
        
        self._vector_store.add_documents(documents)


    def load_pdf_pages(self, file_path: str) -> list[Document]:
        
        reader = pypdf.PdfReader(file_path)
        
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_path, "page": i}
            )
            for i, page in enumerate(reader.pages)
        ]
        
    def text_splitter_and_indexer(self, docs: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        
        all_splits = text_splitter.split_documents(docs)
        
        print(f"Text splitter and indexer: {len(all_splits)}")
        
        self._vector_store.add_documents(all_splits)
        
        print("added document splits to vector store")
    
    def query_vector_store(self) -> tuple[Document, float]:
        results = self._vector_store.similarity_search_with_score(
            #"what are the guardrails of agentic ai"
            query="what are cats like"
        )
        
        print("query result from vector store")
        return results[0]
        
    def _create_retriever(self):
        @chain
        def retriever_chain(query: str) -> list[Document]:
            return self._vector_store.similarity_search(query, k=1)
        return retriever_chain


semantic = Semantic()
semantic.vector_documents()
    
# file_path: str = "D:\\Interview prep\\Agentic AI Books\\State-of-AI-Agents-2026-Final.pdf"
# docs=semantic.load_pdf_pages(file_path)
# print(f"Called load_pdf_pages, len of documents: {len(docs)}")

# semantic.text_splitter_and_indexer(docs)


doc, score = semantic.query_vector_store()
print(doc)
print(f"Score:{score}\n")


# semantic._retriever.batch(
#     [""]
# )

