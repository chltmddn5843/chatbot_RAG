select b.FILE_CHUNK_ID, b.QNA_SEQ , a.answer, a.question, b.SCORE 
  from misobot.tb_qna_log a
     , misobot.tb_context_log b
 where a.qna_seq = b.qna_seq
   and a.question like '%가사근로자가 지원 받을 수 있는 사항에 대해 알려줄래%'
   #and a.QNA_SEQ =
# order by ;
order by a.REG_DT desc, SCORE  desc;
