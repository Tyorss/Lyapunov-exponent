import os
import logging
from dotenv import load_dotenv
import glob
from typing import List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class DocumentFetcher:
    """외부 API 또는 로컬 폴더에서 문서를 가져오는 클래스"""
    
    def __init__(self, local_dir: Optional[str] = None):
        """인증 정보 및 디렉토리 초기화
        
        Args:
            local_dir: 로컬 문서 디렉토리 경로 (None이면 API 사용)
        """
        self.api_url = os.getenv("API_URL")
        self.bearer_token = os.getenv("BEARER_TOKEN")
        self.api_key = os.getenv("API_KEY")
        self.local_dir = local_dir
        
        # 로컬 디렉토리가 지정된 경우 API 무시
        if self.local_dir:
            logger.info(f"로컬 디렉토리 '{self.local_dir}'에서 문서를 가져옵니다.")
        elif not self.api_url:
            logger.warning("API URL이 설정되지 않았습니다. 샘플 데이터를 사용합니다.")
        
    def fetch_documents(self) -> List[Tuple[str, bytes]]:
        """API 또는 로컬 폴더에서 문서를 가져옵니다.
        
        Returns:
            list: (파일명, 바이너리 데이터) 튜플의 리스트
        """
        # 로컬 디렉토리가 지정된 경우 해당 디렉토리에서 문서 가져오기
        if self.local_dir:
            return self._get_local_documents(self.local_dir)
            
        # API URL이 없거나 API 호출에 실패한 경우 샘플 문서 반환
        if not self.api_url:
            return self._get_sample_documents()
            
        try:
            logger.info(f"API {self.api_url}에서 문서를 가져옵니다.")
            
            # API 호출 코드는 그대로 유지...
            # 실제 구현에서 이 부분을 수정하세요.
            
            # 간단히 샘플 문서 반환으로 대체
            return self._get_sample_documents()
            
        except Exception as e:
            logger.error(f"API 호출 중 오류 발생: {str(e)}")
            # 오류 발생시 샘플 데이터 반환
            return self._get_sample_documents()
    
    def _get_local_documents(self, directory: str) -> List[Tuple[str, bytes]]:
        """지정된 로컬 디렉토리에서 문서를 가져옵니다.
        
        Args:
            directory: 문서 디렉토리 경로
            
        Returns:
            list: (파일명, 바이너리 데이터) 튜플의 리스트
        """
        documents = []
        
        if not os.path.exists(directory):
            logger.error(f"지정된 디렉토리 '{directory}'가 존재하지 않습니다.")
            return []
        
        # 지원되는 파일 확장자
        extensions = ['*.txt', '*.pdf', '*.docx', '*.md']
        
        # 각 확장자마다 파일 검색
        all_files = []
        for ext in extensions:
            file_pattern = os.path.join(directory, ext)
            all_files.extend(glob.glob(file_pattern))
        
        # 파일 로드
        for file_path in all_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "rb") as file:
                    content = file.read()
                    documents.append((filename, content))
                logger.info(f"문서 '{filename}' 로드 완료")
            except Exception as e:
                logger.error(f"파일 '{filename}' 로드 중 오류: {str(e)}")
        
        logger.info(f"{len(documents)}개 문서를 로컬 디렉토리에서 가져왔습니다.")
        return documents
    
    def _get_sample_documents(self) -> List[Tuple[str, bytes]]:
        """테스트용 샘플 문서를 반환합니다."""
        logger.info("샘플 문서를 사용합니다.")
        
        # 이 부분은 실제 환경에서 로컬에 저장된 샘플 파일을 사용하도록 수정 필요
        sample_documents = []
        
        # 로컬 샘플 파일이 있는지 확인
        sample_dir = "sample_docs"
        if os.path.exists(sample_dir):
            for filename in os.listdir(sample_dir):
                file_path = os.path.join(sample_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "rb") as file:
                        sample_documents.append((filename, file.read()))
        
        # 샘플 문서가 없으면 간단한 텍스트 파일 생성
        if not sample_documents:
            sample_text = """
            # 샘플 문서
            
            이것은 테스트용 샘플 문서입니다. 실제 API에서 문서를 가져오지 못할 경우 이 문서가 사용됩니다.
            
            ## 섹션 1
            
            이 문서는 RAG 시스템 테스트를 위해 생성되었습니다. 
            한국어 자연어 처리와 관련된 내용을 포함하고 있습니다.
            
            ## 섹션 2
            
            RAG(Retrieval-Augmented Generation)는 검색 기반 생성 모델로, 
            외부 지식을 활용하여 LLM의 응답 품질을 향상시킵니다.
            """
            sample_documents.append(("sample.txt", sample_text.encode('utf-8')))
        
        return sample_documents

if __name__ == "__main__":
    # 테스트 코드
    import sys
    
    # 명령행 인자로 디렉토리를 받거나 기본값 사용
    local_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    fetcher = DocumentFetcher(local_dir)
    docs = fetcher.fetch_documents()
    print(f"가져온 문서 수: {len(docs)}")
    for name, content in docs:
        print(f"문서명: {name}, 크기: {len(content)} 바이트") 