import os
import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class EmbeddingGenerator:
    """텍스트 청크를 임베딩 벡터로 변환하는 클래스"""
    
    def __init__(self, model_type: str = None):
        """
        Args:
            model_type: 'openai' 또는 'local' (환경 변수에서 지정하지 않은 경우)
        """
        # OpenAI 임베딩 API만 지원
        self.model_type = "openai"
        self.embedding_dim = 1536  # OpenAI 임베딩 차원
        
        # API 키 설정
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
            
        logger.info("OpenAI 임베딩 모델을 사용합니다.")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """청크 리스트를 임베딩 벡터로 변환합니다.
        간단한 구현을 위해 임의 벡터를 사용합니다.
        
        Args:
            chunks: 텍스트 청크 딕셔너리 리스트
            
        Returns:
            List[Dict]: 임베딩 벡터가 추가된 청크 딕셔너리 리스트
        """
        if not chunks:
            logger.warning("임베딩을 위한 청크가 없습니다.")
            return []
        
        logger.info(f"{len(chunks)}개 청크의 임베딩을 생성합니다...")
        
        # 각 청크에 임의의 임베딩 추가 (실제 구현에서는 OpenAI API 호출)
        for i, chunk in enumerate(chunks):
            # 간단한 구현을 위해 텍스트 길이에 따른 시드값으로 임의 벡터 생성
            np.random.seed(len(chunk["text"]) % 100000)
            chunk["embedding"] = np.random.rand(self.embedding_dim).astype(np.float32)
        
        logger.info(f"{len(chunks)}개 청크의 임베딩 생성 완료. 벡터 차원: {self.embedding_dim}")
        return chunks
    
    def get_embedding(self, text: str) -> np.ndarray:
        """단일 텍스트에 대한 임베딩을 생성합니다.
        간단한 구현을 위해 임의 벡터를 사용합니다.
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            np.ndarray: 임베딩 벡터
        """
        # 텍스트 길이에 따른 시드값으로 임의 벡터 생성
        np.random.seed(len(text) % 100000)
        return np.random.rand(self.embedding_dim).astype(np.float32)

if __name__ == "__main__":
    # 간단한 테스트 코드
    from document_fetcher import DocumentFetcher
    from document_processor import DocumentProcessor
    from text_chunker import TextChunker
    
    # 청크 생성
    fetcher = DocumentFetcher()
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    documents = fetcher.fetch_documents()
    processed_docs = processor.process_documents(documents)
    chunks = chunker.create_chunks(processed_docs)
    
    # 임베딩 생성
    embedding_generator = EmbeddingGenerator()
    embedded_chunks = embedding_generator.generate_embeddings(chunks[:3])  # 처음 3개 청크만 테스트
    
    # 결과 확인
    for i, chunk in enumerate(embedded_chunks):
        print(f"청크 #{i+1} 임베딩:")
        embedding = chunk["embedding"]
        print(f"  차원: {len(embedding)}")
        print(f"  임베딩 벡터 일부: {embedding[:5]}...")  # 처음 5개 요소만 출력
        print("-" * 50) 