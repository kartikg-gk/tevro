import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_variants import make_pair, oracle, parse_state, validate


@pytest.mark.parametrize('task',['choice','noul','score'])
def test_paired_labels_follow_exactly_one_changed_fact(task):
    for seed in range(40):
        rows,change=make_pair(task,seed)
        assert all(validate(row) for row in rows)
        assert oracle(rows[0]) != oracle(rows[1])
        before,after=(parse_state(row['state']) for row in rows)
        differences=[(code,field) for code in before for field in before[code]
                     if before[code][field]!=after[code][field]]
        assert differences==[(change['entity'],change['attribute'])]
        questions=[next(iter(json.loads(row['questions_json']).values())) for row in rows]
        assert questions[0]['instructions']==questions[1]['instructions']
        assert questions[0].get('criteria')==questions[1].get('criteria')


@pytest.mark.parametrize('task',['choice','noul','score'])
def test_wrong_labels_are_rejected(task):
    rows,_=make_pair(task,21)
    bad=copy.deepcopy(rows[0])
    q=json.loads(bad['questions_json'])
    q['decision']['label']=json.loads(rows[1]['questions_json'])['decision']['label']
    bad['questions_json']=json.dumps(q)
    with pytest.raises(ValueError,match='label'):
        validate(bad)


def test_pair_is_reproducible_and_rejects_unparsed_state():
    assert make_pair('score',7)==make_pair('score',7)
    rows,_=make_pair('noul',7)
    rows[0]['state']+=' This sentence is outside the verified grammar.'
    with pytest.raises(ValueError,match='grammar'):
        validate(rows[0])
