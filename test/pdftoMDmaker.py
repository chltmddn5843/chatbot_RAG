import fitz  # PyMuPDF
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def pdf_to_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    md_text = ""
    
    # 1. 전체 텍스트 추출
    full_content = ""
    for page in doc:
        full_content += page.get_text()

    # 2. 법률명을 대제목(#)으로 변환
    # 첫 줄이나 특정 패턴을 찾아 제목으로 지정
    lines = full_content.split('\n')
    md_text += f"# {lines[0].strip()}\n\n"

    # 3. '제N조' 패턴을 찾아 중제목(##)으로 변환
    # 예: 제1조(목적) -> ## 제1조(목적)
    content_body = '\n'.join(lines[1:])
    processed_body = re.sub(r'(제\d+조\(.*?\))', r'\n## \1\n', content_body)
    
    # 4. '1.', '2.' 패턴을 목록(-)으로 변환하여 가독성 향상
    processed_body = re.sub(r'\n(\d+\.)', r'\n- \1', processed_body)
    
    md_text += processed_body
    return md_text


def chunk_markdown_hierarchically(md_text):
    # 부모 단위: ## (제N조) 기준
    headers_to_split_on = [("##", "Article")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = md_splitter.split_text(md_text)

    # 자식 단위: 각 조문 내부를 문장/항 단위로 더 잘게 쪼갬
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n- ", "\n\n", "\n", ". ", " "]
    )

    hierarchical_data = []
    for i, parent in enumerate(parent_docs):
        # 부모의 메타데이터에 기입된 조항 정보 추출
        article_title = parent.metadata.get("Article", "Unknown")
        
        # 자식 청크 생성
        child_chunks = child_splitter.split_text(parent.page_content)
        
        hierarchical_data.append({
            "parent_id": i,
            "parent_text": f"{article_title}\n{parent.page_content}",
            "article_title": article_title,
            "children": child_chunks
        })
        
    return hierarchical_data