"""
Justi-Q Streamlit 프론트엔드
형사법 RAG 시스템 웹 인터페이스
"""

import sys
import os
sys.path.append("src")

from dotenv import load_dotenv
load_dotenv()

# LangSmith 설정 (Streamlit 임포트 전에 설정)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "justi-q"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

import streamlit as st
from vectorstore import VectorStore
from rag_chain import RAGChain


# 페이지 설정
st.set_page_config(
    page_title="Justi-Q 형사법 AI",
    page_icon="⚖️",
    layout="wide"
)

# 세션 상태 초기화
@st.cache_resource
def load_rag_system():
    """RAG 시스템 로드 (캐싱)"""
    vectorstore = VectorStore(
        collection_name="legal_documents",
        persist_dir="chroma_db"
    )
    rag_chain = RAGChain(vectorstore)
    return rag_chain


def main():
    # 헤더
    st.title("⚖️ Justi-Q 형사법 AI")
    st.markdown("형사법 관련 질문에 판례와 법령을 기반으로 답변해드립니다.")
    st.divider()

    # 사이드바
    with st.sidebar:
        st.header("설정")
        n_results = st.slider("검색 문서 수", min_value=3, max_value=10, value=5)

        st.divider()
        st.header("정보")
        st.markdown("""
        **데이터 출처:**
        - 판례 750건
        - 결정문 294건
        - 법령 898건
        - 해석 58건

        **모델:**
        - 임베딩: multilingual-e5-large
        - LLM: Llama 3.3 70B
        """)

    # RAG 시스템 로드
    try:
        rag = load_rag_system()
    except ValueError as e:
        if "OPENROUTER_API_KEY" in str(e):
            st.error("시스템 로드 실패: OPENROUTER_API_KEY 환경변수를 설정해주세요.")
            st.info("**Streamlit Cloud 배포 시:** Settings > Secrets에서 `OPENROUTER_API_KEY = \"your_key\"` 추가")
        else:
            st.error(f"시스템 로드 실패: {e}")
        return
    except Exception as e:
        error_msg = str(e)
        st.error(f"시스템 로드 실패: {error_msg}")
        st.info("먼저 `python main.py --index` 로 인덱싱을 실행해주세요.")
        return

    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 참고 문서"):
                    for src in message["sources"]:
                        st.markdown(f"- **[{src['type']}]** {src['doc_id']}")

    # 사용자 입력
    if prompt := st.chat_input("형사법 관련 질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                result = rag.query(prompt, n_results=n_results)

            st.markdown(result["answer"])

            # 참고 문서 표시
            with st.expander("📚 참고 문서"):
                for src in result["sources"]:
                    st.markdown(f"- **[{src['type']}]** {src['doc_id']} (유사도: {1 - src['distance']:.2%})")

        # 어시스턴트 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })


if __name__ == "__main__":
    main()
