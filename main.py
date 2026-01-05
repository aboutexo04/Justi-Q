"""
Justi-Q: 형사법 RAG 시스템
메인 실행 파일
"""

import sys
sys.path.append("src")

from data_loader import LegalDataLoader
from vectorstore import VectorStore
from rag_chain import RAGChain


class JustiQ:
    """형사법 RAG 시스템 메인 클래스"""

    def __init__(
        self,
        data_dir: str = "data_sampled",
        collection_name: str = "legal_documents",
        persist_dir: str = "chroma_db",
        model: str = "meta-llama/llama-3.3-70b-instruct:free"
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            collection_name: ChromaDB 컬렉션 이름
            persist_dir: ChromaDB 저장 경로
            model: OpenRouter LLM 모델명
        """
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.model = model

        self.loader = None
        self.vectorstore = None
        self.rag_chain = None

    def index(self, chunk_size: int = 1000, overlap: int = 200) -> dict:
        """
        데이터 인덱싱: 데이터 로드 → 청킹 → 벡터 DB 저장

        Args:
            chunk_size: 청크 크기
            overlap: 청크 간 오버랩

        Returns:
            인덱싱 결과 통계
        """
        print("=" * 60)
        print("데이터 인덱싱 시작")
        print("=" * 60)

        # 데이터 로드 및 청킹
        self.loader = LegalDataLoader(self.data_dir)
        chunks = self.loader.load_and_chunk(chunk_size=chunk_size, overlap=overlap)

        # 벡터 스토어 생성 및 저장
        self.vectorstore = VectorStore(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir
        )
        self.vectorstore.add_documents(chunks)

        stats = self.vectorstore.get_stats()
        print("\n인덱싱 완료!")
        return stats

    def load(self) -> None:
        """기존 벡터 스토어 로드"""
        self.vectorstore = VectorStore(
            collection_name=self.collection_name,
            persist_dir=self.persist_dir
        )
        self.rag_chain = RAGChain(
            vectorstore=self.vectorstore,
            model=self.model
        )
        print(f"벡터 스토어 로드 완료 (문서 수: {self.vectorstore.collection.count()})")

    def query(self, question: str, n_results: int = 5) -> dict:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문
            n_results: 검색할 문서 수

        Returns:
            {
                "answer": 답변 텍스트,
                "sources": 참고 문서 리스트,
                "question": 원본 질문
            }
        """
        if self.rag_chain is None:
            self.load()

        return self.rag_chain.query(question=question, n_results=n_results)

    def search(self, query: str, n_results: int = 5) -> list:
        """
        관련 문서 검색 (답변 생성 없이)

        Args:
            query: 검색 쿼리
            n_results: 검색 결과 수

        Returns:
            검색된 문서 리스트
        """
        if self.vectorstore is None:
            self.load()

        return self.vectorstore.search(query=query, n_results=n_results)

    def interactive(self) -> None:
        """대화형 모드 실행"""
        if self.rag_chain is None:
            self.load()

        print("\n" + "=" * 60)
        print("Justi-Q 형사법 RAG 시스템")
        print("종료: 'quit' 또는 'q'")
        print("=" * 60)

        while True:
            question = input("\n질문: ").strip()

            if question.lower() in ['quit', 'q', '종료']:
                print("종료합니다.")
                break

            if not question:
                continue

            print("\n답변 생성 중...")
            result = self.query(question)

            print(f"\n{result['answer']}")
            print("\n[참고 문서]")
            for src in result["sources"]:
                print(f"  - [{src['type']}] {src['doc_id']}")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Justi-Q 형사법 RAG 시스템")
    parser.add_argument("--index", action="store_true", help="데이터 인덱싱 실행")
    parser.add_argument("--query", type=str, help="단일 질문 실행")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    parser.add_argument("--data-dir", type=str, default="data_sampled", help="데이터 디렉토리")

    args = parser.parse_args()

    # JustiQ 인스턴스 생성
    justiq = JustiQ(data_dir=args.data_dir)

    if args.index:
        # 인덱싱 모드
        justiq.index()

    elif args.query:
        # 단일 질문 모드
        result = justiq.query(args.query)
        print(f"\n질문: {args.query}")
        print(f"\n답변:\n{result['answer']}")
        print("\n[참고 문서]")
        for src in result["sources"]:
            print(f"  - [{src['type']}] {src['doc_id']}")

    else:
        # 기본: 대화형 모드
        justiq.interactive()


if __name__ == "__main__":
    main()
