# Standard libs
from abc import ABC, abstractmethod


class BaseIndexer(ABC):
    @abstractmethod
    def add(self, embeddings, ids, metadatas=None): ...
    
    @abstractmethod
    def search(self, query_embeddings, n_results=10): ...