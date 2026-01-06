"""
RAG 체인
- OpenRouter를 통한 LLM 연결
- 검색 + 답변 생성 파이프라인
- LangSmith 자동 트레이싱
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable

from vectorstore import VectorStore

load_dotenv()

# LangSmith 설정
def setup_langsmith():
    """LangSmith 트레이싱 설정"""
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "justi-q"  # 강제 설정
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        return True
    return False

LANGSMITH_ENABLED = setup_langsmith()


def get_api_key() -> str:
    """API 키 가져오기 (환경변수 또는 Streamlit secrets)"""
    # 1. 환경변수 확인 (로컬 개발용) - 우선
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # 2. Streamlit secrets 확인 (Streamlit Cloud 배포용)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENROUTER_API_KEY' in st.secrets:
            return st.secrets['OPENROUTER_API_KEY']
    except (ImportError, Exception):
        pass

    raise ValueError("OPENROUTER_API_KEY 환경변수를 설정해주세요.")


class RAGChain:
    """RAG 체인: 검색 + 답변 생성 (LangSmith 자동 트레이싱)"""

    def __init__(
        self,
        vectorstore: VectorStore,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        temperature: float = 0.7
    ):
        self.vectorstore = vectorstore
        self.model = model
        self.temperature = temperature

        # LangChain ChatOpenAI (OpenRouter 연결) - 자동 트레이싱 지원
        api_key = get_api_key()

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=2000,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        self.system_prompt = """당신은 형사법 전문 법률 AI 어시스턴트입니다.
주어진 법률 문서(판례, 법령, 결정문, 해석)를 참고하여 사용자의 질문에 정확하고 전문적으로 답변해주세요.

답변 시 주의사항:
1. 반드시 제공된 문서 내용을 근거로 답변하세요.
2. 관련 법령 조항이 있다면 명시해주세요.
3. 판례가 있다면 사건번호와 함께 인용해주세요.
4. 확실하지 않은 내용은 추측하지 마세요.
5. 답변은 명확하고 이해하기 쉽게 작성해주세요.
"""

    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """검색 결과를 컨텍스트로 포맷팅"""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            metadata = result["metadata"]
            type_name = metadata.get("type_name", "문서")
            doc_id = metadata.get("doc_id", "unknown")

            context_parts.append(
                f"[문서 {i}] ({type_name}) - {doc_id}\n{result['content']}\n"
            )

        return "\n---\n".join(context_parts)

    @traceable(name="rag_query")
    def query(
        self,
        question: str,
        n_results: int = 5,
        filter_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        # 1. 관련 문서 검색
        search_results = self.vectorstore.search(
            query=question,
            n_results=n_results,
            filter_type=filter_type
        )

        if not search_results:
            return {
                "answer": "관련 문서를 찾을 수 없습니다.",
                "sources": [],
                "question": question
            }

        # 2. 컨텍스트 구성
        context = self._format_context(search_results)

        # 3. LLM 호출 (LangChain - 자동 트레이싱)
        user_message = f"""다음은 관련 법률 문서입니다:

{context}

---

질문: {question}

위 문서를 참고하여 답변해주세요."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        # 4. 결과 반환
        sources = [
            {
                "doc_id": r["metadata"]["doc_id"],
                "type": r["metadata"]["type_name"],
                "distance": r["distance"]
            }
            for r in search_results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "question": question
        }


def main():
    """RAG 체인 테스트"""
    # 벡터 스토어 로드 (이미 인덱싱된 경우)
    vectorstore = VectorStore()

    # RAG 체인 생성
    rag = RAGChain(vectorstore)

    # 테스트 질문
    test_questions = [
        "폭행죄의 처벌 기준은 어떻게 되나요?",
        "음주운전 처벌 규정을 알려주세요.",
        "사기죄 성립 요건은 무엇인가요?"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"질문: {question}")
        print("="*60)

        result = rag.query(question)

        print(f"\n답변:\n{result['answer']}")
        print(f"\n참고 문서:")
        for src in result["sources"]:
            print(f"  - [{src['type']}] {src['doc_id']} (거리: {src['distance']:.4f})")


if __name__ == "__main__":
    main()
