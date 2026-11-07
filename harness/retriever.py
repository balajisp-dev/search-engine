"""
Retriever Protocol
"""
from typing import Protocol,runtime_checkable

@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self ,docs: dict[str,str])->None:
        """Index the documents."""
        ...

    def search(self,query: str,k: int)->list[tuple[str,float]]:
        """Search the index for the top k results."""
        ...
