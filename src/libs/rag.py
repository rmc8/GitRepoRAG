import os

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class Rag:
    def __init__(self, settings: dict):
        self.settings = settings

    def file_filter(self, file_path: str):
        for file_type in self.settings["file_filter"]:
            if file_path.endswith(file_type):
                return True
        return False

    @staticmethod
    def get_embeddings() -> OllamaEmbeddings:
        model: str = os.getenv("ollama_embedding_model")
        base_url: str = os.getenv("ollama_base_url")
        embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )
        return embeddings

    @staticmethod
    def get_model() -> ChatOllama:
        model: str = os.getenv("ollama_model")
        base_url: str = os.getenv("ollama_base_url")
        embeddings = ChatOllama(
            model=model,
            base_url=base_url,
        )
        return embeddings

    def get_vec_db(self):
        loader = GitLoader(
            clone_url=os.getenv("repo_url"),
            repo_path=os.getenv("repo_path"),
            file_filter=self.file_filter,
            branch=os.getenv("branch"),
        )
        documents = loader.load()
        embeddings = self.get_embeddings()
        db = Chroma.from_documents(documents, embeddings)
        return db

    def run(self):
        model = self.get_model()
        db = self.get_vec_db()
        retriever = db.as_retriever()
        prompt = ChatPromptTemplate.from_template(
            """
            以下の文脈だけを踏まえて質問に回答してください。
            文脈: '''
            {context}
            '''
            質問: {question}
            """.strip()
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": retriever,
            }
            | prompt
            | model
            | StrOutputParser()
        )
        while True:
            req = input("Please enter your question >> ")
            print()
            res = chain.invoke(req)
            print(res)
            print("\n\n")
