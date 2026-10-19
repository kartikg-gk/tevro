"""Easy paired variations using the same item/shelf/weight tasks as synthetic.py.

These are newly generated procedural cases, not rewritten source records.
Labels are checked from the rendered state by a separate parser/oracle.
"""
import copy
import random
import re

COLORS = ['red', 'blue', 'green', 'black', 'white', 'orange']
SIZES = ['small', 'medium', 'large']
LEVELS = ['light (under 5 kg)', 'medium (5 to 20 kg)', 'heavy (over 20 kg)']
ITEM = re.compile(r'Item ([A-Z]{2}-\d{3}) is on shelf ([A-H][1-9])\. It is (red|blue|green|black|white|orange) and (small|medium|large), weighs (\d+) kg and is (not fragile|fragile)\.')


def render(items):
    return ' '.join(f"Item {i['code']} is on shelf {i['shelf']}. It is {i['color']} and {i['size']}, weighs {i['kg']} kg and is {'fragile' if i['fragile'] else 'not fragile'}." for i in items)


def parse_state(state):
    matches = list(ITEM.finditer(state))
    if not matches or ' '.join(m.group(0) for m in matches) != state:
        raise ValueError('State does not match the controlled grammar')
    items = {}
    for match in matches:
        code, shelf, color, size, kg, fragile = match.groups()
        if code in items:
            raise ValueError('Duplicate entity code')
        items[code] = dict(code=code, shelf=shelf, color=color, size=size, kg=int(kg), fragile=fragile=='fragile')
    return items


def oracle(row):
    """Derive the answer from serialized inputs, never the stored generation label."""
    items = parse_state(row['state'])
    questions = __import__('json').loads(row['questions_json'])
    if len(questions) != 1:
        raise ValueError('Expected exactly one generated question per row')
    q = next(iter(questions.values()))
    text = q['instructions']
    if q['type'] == 'choice':
        match = re.fullmatch(r'Which shelf holds item ([A-Z]{2}-\d{3})\?', text)
        if not match:
            raise ValueError('Unsupported choice question')
        correct = items[match[1]]['shelf']
        keys = [k for k,v in q['criteria'].items() if v==correct]
        if len(keys)!=1:
            raise ValueError('Answer is not unique')
        return keys[0]
    if q['type'] == 'noul':
        match = re.fullmatch(r'Is item ([A-Z]{2}-\d{3}) (not )?fragile\?', text)
        if not match:
            raise ValueError('Unsupported yes/no question')
        return items[match[1]]['fragile'] != bool(match[2])
    if q['type'] == 'score':
        match = re.fullmatch(r'How heavy is item ([A-Z]{2}-\d{3})\?', text)
        if not match or q['criteria'] != LEVELS:
            raise ValueError('Unsupported scoring rubric')
        kg = items[match[1]]['kg']
        return 0 if kg<5 else 1 if kg<=20 else 2
    raise ValueError('Unknown question type')


def validate(row):
    import json
    q = next(iter(json.loads(row['questions_json']).values()))
    answer = oracle(row)
    if type(answer) is not type(q['label']) or answer != q['label']:
        raise ValueError('Stored label disagrees with the rendered-state oracle')
    return True


def make_pair(task, seed):
    import json
    rng = random.Random(seed)
    n = rng.randint(3,5)
    codes = rng.sample([a+b for a in 'ABCDEFGHJKLMNPQRSTUVWXYZ' for b in 'ABCDEFGHJKLMNPQRSTUVWXYZ'],n)
    shelves = rng.sample([a+str(b) for a in 'ABCDEFGH' for b in range(1,10)],n)
    items = [dict(code=c+f'-{rng.randrange(100,1000)}',shelf=s,color=rng.choice(COLORS),
                  size=rng.choice(SIZES),kg=rng.randint(1,40),fragile=bool(rng.getrandbits(1)))
             for c,s in zip(codes,shelves)]
    target = rng.randrange(n)
    q = {'type':task}
    if task == 'choice':
        options = shelves[:]
        rng.shuffle(options)
        q.update(instructions=f"Which shelf holds item {items[target]['code']}?",
                 criteria={f'option_{j}':s for j,s in enumerate(options)})
        attribute = 'shelf'
        replacement = rng.choice([s for s in shelves if s!=items[target]['shelf']])
    elif task == 'noul':
        negated = bool(rng.getrandbits(1))
        q['instructions'] = f"Is item {items[target]['code']} {'not ' if negated else ''}fragile?"
        attribute = 'fragile'
        replacement = not items[target]['fragile']
    elif task == 'score':
        q.update(instructions=f"How heavy is item {items[target]['code']}?",criteria=LEVELS[:])
        items[target]['kg'], replacement = rng.choice([(4,5),(5,4),(20,21),(21,20)])
        attribute = 'kg'
    else:
        raise ValueError(task)
    edited = copy.deepcopy(items)
    edited[target][attribute] = replacement
    rows=[]
    for variant, world in enumerate((items,edited)):
        row = dict(state=render(world),questions_json=json.dumps({'decision':q},ensure_ascii=False),
                   domain='Structured rule following',question_types=[task],question_count=1)
        labelled = copy.deepcopy(q)
        labelled['label'] = oracle(row)
        row['questions_json']=json.dumps({'decision':labelled},ensure_ascii=False,separators=(',',':'))
        validate(row)
        rows.append(row)
    assert oracle(rows[0]) != oracle(rows[1])
    assert sum(a[k]!=b[k] for a,b in zip(items,edited) for k in a)==1
    return rows, dict(task=task,seed=seed,entity=items[target]['code'],attribute=attribute,
                     before=items[target][attribute],after=replacement)
