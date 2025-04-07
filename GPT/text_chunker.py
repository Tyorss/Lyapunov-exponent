import re
import logging
from typing import List, Dict, Any, Optional, Union

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextChunker:
    """문서 텍스트를 청크로 분할하는 클래스"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, use_tiktoken: bool = False):
        """
        Args:
            chunk_size: 청크 단위 크기 (글자수)
            chunk_overlap: 청크 간 중복 크기 (글자수)
            use_tiktoken: 사용하지 않음 (호환성을 위해 파라미터 유지)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_tiktoken = False  # 항상 False로 설정 (tiktoken 사용 안함)
        logger.info(f"청크 크기: {chunk_size}, 중복 크기: {chunk_overlap}, 글자 수 기준 청킹 사용")
    
    def create_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서들을 처리하여 청크를 생성합니다.
        
        Args:
            documents: 텍스트 필드를 포함한 문서 딕셔너리 리스트
            
        Returns:
            List[Dict]: 생성된 청크들의 리스트
        """
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            all_chunks.extend(doc_chunks)
            logger.info(f"'{doc['filename']}' 문서를 {len(doc_chunks)}개 청크로 분할했습니다.")
        
        logger.info(f"총 {len(all_chunks)}개 청크가 생성되었습니다.")
        return all_chunks
    
    def _chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """단일 문서를 청크로 분할합니다.
        
        Args:
            document: 텍스트 필드를 포함한 문서 딕셔너리
            
        Returns:
            List[Dict]: 문서에서 생성된 청크들의 리스트
        """
        text = document.get("text", "")
        if not text or (document.get("error", False)):
            # 오류가 있는 문서는 하나의 청크로 유지
            return [document]
            
        # 텍스트 분할 방식 결정
        chunks = []
        
        # 구조 기반 청킹: 문단 단위로 분할
        paragraphs = self._split_into_paragraphs(text)
        
        # 청킹 적용 (항상 글자 수 기준 청킹 사용)
        chunks = self._chunk_by_chars(paragraphs, document)
            
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """텍스트를 문단 단위로 분할합니다."""
        # 빈 줄을 기준으로 문단 구분
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 빈 문단 제거 및 공백 정리
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _chunk_by_chars(self, paragraphs: List[str], document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """문자 수 기준으로 청킹합니다."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # 현재 문단을 추가해도 크기가 넘지 않으면 추가
            para_len = len(paragraph)
            
            if current_size + para_len <= self.chunk_size:
                current_chunk.append(paragraph)
                current_size += para_len
            else:
                # 현재 chunk가 있으면 저장
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk_dict(chunk_text, document))
                
                # 새 청크 시작
                current_chunk = [paragraph]
                current_size = para_len
        
        # 마지막 청크 저장
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk_dict(chunk_text, document))
        
        # 중복 적용
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap_chars(chunks)
        
        return chunks
    
    def _apply_overlap_chars(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문자 기준 중복 적용"""
        result_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                # 이전 청크의 마지막 부분을 현재 청크 앞에 추가
                prev_text = chunks[i-1]["text"]
                
                if len(prev_text) > self.chunk_overlap:
                    overlap_text = prev_text[-self.chunk_overlap:]
                    new_text = overlap_text + chunk["text"]
                    
                    chunk_copy = chunk.copy()
                    chunk_copy["text"] = new_text
                    result_chunks.append(chunk_copy)
                    continue
            
            result_chunks.append(chunk)
        
        return result_chunks
    
    def _create_chunk_dict(self, text: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """청크 딕셔너리를 생성합니다."""
        # 메타데이터는 원본 문서에서 복사
        metadata = document.get("metadata", {}).copy()
        
        return {
            "text": text,
            "metadata": metadata,
            "source_doc": document["filename"],
            "chunk_size": len(text)
        }

if __name__ == "__main__":
    # 간단한 테스트 코드
    from document_fetcher import DocumentFetcher
    from document_processor import DocumentProcessor
    
    fetcher = DocumentFetcher()
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    # 문서 가져오기
    documents = fetcher.fetch_documents()
    
    # 문서 처리
    processed_docs = processor.process_documents(documents)
    
    # 청킹
    chunks = chunker.create_chunks(processed_docs)
    
    # 결과 확인
    for i, chunk in enumerate(chunks):
        print(f"청크 #{i+1}")
        print(f"크기: {chunk.get('chunk_size', '?')} 자")
        print(f"출처: {chunk['source_doc']}")
        print(f"내용 미리보기: {chunk['text'][:100]}...")
        print("-" * 50) 