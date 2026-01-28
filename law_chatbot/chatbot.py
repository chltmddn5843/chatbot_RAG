import os
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv

#db_utils에서 필요한 함수들을 불러옵니다.
from db_utils import ensure_chatbot_logs_table, save_chat_log, show_recent_logs

# 초기 설정
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # DLL 에러 방지를 위한 CPU 강제 모드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

client = OpenAI(api_key=os.getenv("API_KEY"))
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_COLLECTION = "col_1"

# 모델 로드 (HuggingFace)
device = "cpu"
model_name = "nomic-ai/nomic-embed-text-v2-moe"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

def get_embedding(text):
    input_text = f"search_query: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        mean_pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embedding = mean_pooled[0].cpu().numpy()
    return embedding.tolist()

def search_milvus(query, top_k=5):
    if not connections.has_connection("default"):
        connections.connect(alias="default", host=MILVUS_HOST, port=19530)
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    query_vector = get_embedding(query)
    
    results = collection.search(
        data=[query_vector],
        anns_field="dense",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["pk", "source", "file_hash", "page", "row"]
    )
    
    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id": hit.entity.get("pk"),
            "content": hit.entity.get("source"),
            "parent_id": hit.entity.get("file_hash"),
            "page": hit.entity.get("page"),
            "score": hit.distance
        })
    return hits

# 6. 답변 생성
def ask_legal_expert(user_input, retrieved_chunks):
    context_text = "\n\n".join([f"근거: {c['content']}" for c in retrieved_chunks])
    
    system_prompt = """당신은 법률 지식을 전달하는 **'법률 데이터 검증 전문가'**입니다. 
당신의 목적은 사용자에게 정확한 법 정보를 제공함과 동시에, RAG 시스템의 성능 측정을 위해 답변의 근거가 된 컨텍스트를 구조화하여 출력하는 것입니다.

[계층 구조 설계]
- Level 1 (#): 법률명
- Level 3 (###): '조(Article)'
- Level 4 (####): '항(Paragraph)'

[응답 구조 가이드라인]
1. [최종 답변]: 법적 근거를 바탕으로 설명 (표/리스트 활용)
2. [참조 조문]: 구체적인 조항 번호 명시
3. [Ragas Evaluation Data]: JSON 형식 데이터 포함 (question, contexts, answer, ground_truth)
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[Context]\n{context_text}\n\n[Question]\n{user_input}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    # 프로그램 시작 시 DB 테이블이 없으면 생성
    ensure_chatbot_logs_table()
    print("⚖️ 법률 RAG 시스템이 가동되었습니다.")
    
    while True:
        query = input("\n질문을 입력하세요 (exit: 종료, logs: 로그확인): ")
        if query.lower() == "exit": break
        if query.lower() == "logs":
            show_recent_logs()
            continue
        
        try:
            # 1. Milvus 검색
            hits = search_milvus(query, top_k=5)
            # 2. 답변 생성
            answer = ask_legal_expert(query, hits)
            # 3. DB 로깅 (import한 함수 사용)
            save_chat_log(query, answer, hits)
            
            print("\n" + "-"*30 + " AI 응답 " + "-"*30)
            print(answer)
        except Exception as e:
            print(f"❌ 오류: {e}")