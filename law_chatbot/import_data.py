"""Load the repository's precomputed legal vectors into an empty local Milvus."""
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    rows = [row for path in sorted((root / 'law_chuunking_test/json_result').glob('*.json'))
            for row in json.loads(path.read_text())]
    if not rows:
        raise ValueError('적재할 JSON 데이터가 없습니다.')
    for row in rows:
        if len(row['dense']) != 768 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row['dense']):
            raise ValueError(f"잘못된 벡터: {row['pk']}")
    if len({r['pk'] for r in rows}) != len(rows):
        raise ValueError('중복 PK가 있습니다.')
    connections.connect(host=os.getenv('MILVUS_HOST', '127.0.0.1'), port=19530, timeout=5)
    name = os.getenv('MILVUS_COLLECTION', 'col_1')
    if utility.has_collection(name):
        raise RuntimeError(f'{name}이 이미 존재합니다. 기존 데이터는 변경하지 않았습니다.')
    fields = [FieldSchema(name='pk', dtype=DataType.VARCHAR, max_length=512, is_primary=True),
              FieldSchema(name='dense', dtype=DataType.FLOAT_VECTOR, dim=768)]
    fields += [FieldSchema(name=k, dtype=DataType.VARCHAR, max_length=65535) for k in ('source', 'text', 'file_hash')]
    fields += [FieldSchema(name=k, dtype=DataType.INT64) for k in ('page', 'row')]
    collection = Collection(name, CollectionSchema(fields))
    collection.insert(rows)
    collection.flush()
    collection.create_index('dense', {'index_type': 'FLAT', 'metric_type': 'L2', 'params': {}})
    collection.load()
    print(f'{name}: {collection.num_entities}개 적재 완료')


if __name__ == '__main__':
    main()
