"""
벡터 스토어 관리
- multilingual-e5-large 임베딩
- ChromaDB 벡터 저장소
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def get_device() -> str:
    """사용 가능한 디바이스 자동 감지 (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class VectorStore:
    """ChromaDB 기반 벡터 스토어"""

    def __init__(
        self,
        collection_name: str = "legal_documents",
        persist_dir: str = "chroma_db",
        embedding_model: str = "intfloat/multilingual-e5-large"
    ):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        # 디바이스 설정
        self.device = get_device()
        print(f"사용 디바이스: {self.device}")

        # 임베딩 모델 로드
        print(f"임베딩 모델 로드 중: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        print("임베딩 모델 로드 완료")

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # 컬렉션 가져오기 또는 생성
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"컬렉션 '{collection_name}' 준비 완료 (문서 수: {self.collection.count()})")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩 생성 (E5 모델용 prefix 추가)"""
        # E5 모델은 "passage: " prefix 필요
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self.embedding_model.encode(
            prefixed_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 임베딩 생성 (E5 모델용 prefix 추가)"""
        # E5 모델은 쿼리에 "query: " prefix 필요
        prefixed_query = f"query: {query}"
        embedding = self.embedding_model.encode(
            prefixed_query,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """문서 청크를 벡터 스토어에 추가"""
        print(f"\n{len(chunks)}개 청크를 벡터 스토어에 추가 중...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="임베딩"):
            batch = chunks[i:i + batch_size]

            ids = [chunk["metadata"]["chunk_id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]

            # 임베딩 생성
            embeddings = self._get_embeddings(documents)

            # ChromaDB에 추가
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

        print(f"벡터 스토어 저장 완료 (총 문서 수: {self.collection.count()})")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """유사 문서 검색"""
        # 쿼리 임베딩
        query_embedding = self._get_query_embedding(query)

        # 필터 설정
        where_filter = None
        if filter_type:
            where_filter = {"type": filter_type}

        # 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # 결과 정리
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return search_results

    def get_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_dir": str(self.persist_dir)
        }


if __name__ == "__main__":
    # 테스트
    from data_loader import LegalDataLoader

    # 데이터 로드
    loader = LegalDataLoader("data_sampled")
    chunks = loader.load_and_chunk(chunk_size=1000, overlap=200)

    # 벡터 스토어 생성 및 문서 추가
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks)

    # 검색 테스트
    print("\n=== 검색 테스트 ===")
    query = "폭행죄 처벌 기준"
    results = vectorstore.search(query, n_results=3)

    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result['metadata']['type_name']} - {result['metadata']['doc_id']}")
        print(f"    거리: {result['distance']:.4f}")
        print(f"    내용: {result['content'][:150]}...")
