import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_local_question_file_preserves_choices_and_ordinal_labels(tmp_path):
    import pyarrow.parquet as pq
    import pyarrow as pa
    from data_variants import make_pair
    from dataset import load_questions
    raw=[]
    for task in ('choice','noul','score'):
        raw.extend(make_pair(task,13)[0])
    path=tmp_path/'questions.parquet'
    pq.write_table(pa.Table.from_pylist(raw),path)
    loaded=load_questions('train',None,0,{'train':str(path)})
    expected={(r['state'],q['instructions']):q for r in raw
              for q in json.loads(r['questions_json']).values()}
    assert len(loaded)==6
    for row in loaded:
        q=expected[row['state'],row['instructions']]
        if q['type']=='choice':
            assert row['label']==list(q['criteria']).index(q['label'])
        else:
            assert row['label']==int(q['label'])
        assert len(row['choices'])>=2
