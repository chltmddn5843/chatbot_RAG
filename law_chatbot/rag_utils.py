def build_rag_prompt(user_input, retrieved_chunks):
    context = "\n\n".join([f"근거: {c['content']}" for c in retrieved_chunks])
    prompt = f"다음 질문에 답변하세요.\n질문: {user_input}\n\n{context}\n답변:"
    return prompt
