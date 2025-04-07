import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS를 사용한 벡터 저장소 및 검색 클래스"""
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: 임베딩 벡터의 차원 수
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []  # 원본 청크 정보를 저장할 리스트
        
        # L2 거리 기반 기본 인덱스 생성
        self._create_index()
        
        logger.info(f"{embedding_dim} 차원의 새 벡터 인덱스가 초기화되었습니다.")
    
    def _create_index(self):
        """FAISS 인덱스를 생성합니다."""
        # 기본 L2 거리 기반 인덱스 (정확도 우선)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_embeddings(self, embedded_chunks: List[Dict[str, Any]]):
        """임베딩된 청크들을 인덱스에 추가합니다.
        
        Args:
            embedded_chunks: 임베딩 벡터가 포함된 청크 딕셔너리 리스트
        """
        if not embedded_chunks:
            logger.warning("추가할 임베딩이 없습니다.")
            return
        
        # 임베딩 벡터 추출 및 형식 변환
        embeddings = []
        for chunk in embedded_chunks:
            # 임베딩 필드가 있는지 확인
            if "embedding" not in chunk:
                logger.warning(f"청크에 임베딩이 없습니다: {chunk.get('text', '')[:50]}...")
                continue
            
            # 임베딩 추출
            embedding = chunk["embedding"]
            embeddings.append(embedding)
            
            # 청크 저장 (임베딩은 제외하고 저장하여 메모리 절약)
            chunk_copy = chunk.copy()
            chunk_copy.pop("embedding", None)
            self.chunks.append(chunk_copy)
        
        if not embeddings:
            logger.warning("유효한 임베딩이 없습니다.")
            return
        
        # numpy 배열로 변환
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # 인덱스에 추가
        self.index.add(embeddings_np)
        
        logger.info(f"{len(embeddings)} 개의 임베딩이 인덱스에 추가되었습니다. 현재 총 {self.index.ntotal}개.")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """쿼리 임베딩과 가장 유사한 청크를 검색합니다.
        
        Args:
            query_embedding: 쿼리 텍스트의 임베딩 벡터
            top_k: 반환할 최대 결과 수
            
        Returns:
            List[Dict]: 검색 결과 청크 딕셔너리 리스트 (거리 정보 포함)
        """
        if self.index.ntotal == 0:
            logger.warning("인덱스가 비어있어 검색할 수 없습니다.")
            return []
        
        # 쿼리 벡터 형태 확인 및 변환
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # FAISS로 검색 수행
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # 결과 가공
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue  # 유효하지 않은 인덱스 무시
                
            # 원본 청크 정보 가져오기
            chunk = self.chunks[idx]
            
            # 거리 정보 추가
            result = chunk.copy()
            result["distance"] = float(dist)
            result["score"] = self._convert_distance_to_score(dist)
            result["rank"] = i + 1
            
            results.append(result)
        
        logger.info(f"쿼리에 대해 {len(results)}개의 관련 청크를 찾았습니다.")
        return results
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """L2 거리를 유사도 점수(0~1)로 변환합니다.
        
        거리가 작을수록 유사도가 높으므로 변환이 필요합니다.
        
        Args:
            distance: L2 거리
            
        Returns:
            float: 유사도 점수 (0~1, 1이 가장 유사함)
        """
        # 거리의 최대값 (경험적으로 설정)
        max_distance = 100.0
        
        # 거리를 0~1 범위의 스코어로 변환 (1이 가장 유사함)
        score = max(0.0, min(1.0, 1.0 - (distance / max_distance)))
        return score
    
    def save(self, file_path: str):
        """벡터 인덱스를 파일로 저장합니다.
        
        Args:
            file_path: 저장할 파일 경로
        """
        try:
            # 인덱스 파일 저장
            faiss.write_index(self.index, f"{file_path}.index")
            
            # 청크 정보 저장
            with open(f"{file_path}.chunks", 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"벡터 인덱스가 {file_path} 파일에 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"벡터 인덱스 저장 중 오류 발생: {str(e)}")
    
    @classmethod
    def load(cls, file_path: str, embedding_dim: int) -> 'VectorStore':
        """저장된 벡터 인덱스를 로드합니다.
        
        Args:
            file_path: 인덱스 파일 경로
            embedding_dim: 임베딩 벡터의 차원 수
            
        Returns:
            VectorStore: 로드된 벡터 저장소 인스턴스
        """
        try:
            # 빈 인스턴스 생성
            instance = cls(embedding_dim)
            
            # 인덱스 로드
            instance.index = faiss.read_index(f"{file_path}.index")
            
            # 청크 정보 로드
            with open(f"{file_path}.chunks", 'rb') as f:
                instance.chunks = pickle.load(f)
            
            logger.info(f"벡터 인덱스가 {file_path}에서 로드되었습니다. 총 {instance.index.ntotal}개의 벡터.")
            return instance
            
        except Exception as e:
            logger.error(f"벡터 인덱스 로드 중 오류 발생: {str(e)}")
            # 문제 발생 시 새 인스턴스 반환
            return cls(embedding_dim)

if __name__ == "__main__":
    # 간단한 테스트 코드
    from document_fetcher import DocumentFetcher
    from document_processor import DocumentProcessor
    from text_chunker import TextChunker
    from embedding_generator import EmbeddingGenerator
    
    # 전체 파이프라인 테스트
    fetcher = DocumentFetcher()
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    # 문서 처리
    documents = fetcher.fetch_documents()
    processed_docs = processor.process_documents(documents)
    chunks = chunker.create_chunks(processed_docs)
    
    # 임베딩 생성
    embedding_generator = EmbeddingGenerator()
    embedded_chunks = embedding_generator.generate_embeddings(chunks)
    
    # 벡터 인덱스에 추가
    vector_store = VectorStore(embedding_generator.embedding_dim)
    vector_store.add_embeddings(embedded_chunks)
    
    # 검색 테스트
    test_query = "RAG 시스템이란 무엇인가요?"
    query_embedding = embedding_generator.get_embedding(test_query)
    
    results = vector_store.search(query_embedding, top_k=3)
    
    # 결과 출력
    print(f"쿼리: {test_query}")
    print(f"검색 결과 ({len(results)}개):")
    for i, result in enumerate(results):
        print(f"[{i+1}] 스코어: {result['score']:.4f}, 거리: {result['distance']:.4f}")
        print(f"    출처: {result['source_doc']}")
        print(f"    내용: {result['text'][:100]}...")
        print("-" * 50) 