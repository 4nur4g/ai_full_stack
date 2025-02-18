from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction, Documents


class LangChainEmbeddingAdapter(EmbeddingFunction[Documents]):
    def __init__(self, ef: Embeddings):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        # LC EFs also have embed_query but Chroma doesn't support that so we just use embed_documents
        # TODO: better type checking
        return self.ef.embed_documents(input)
