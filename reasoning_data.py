"""Procedural rule-reasoning questions: eligibility under conditions and exceptions, payouts under deductibles and
limits, and counting rule violations. Every answer is computed from the generated facts, never from a model.

Each row matches the dataset schema: state, questions_json (one question), domain, question_types, question_count.
"""
import json
import random

TOPICS = {
    "refund": {"action": "issue the refund", "subject": "customer",
               "conditions": [("has the original receipt", "has no receipt"),
                              ("returned the item unopened", "opened the item"),
                              ("is a registered member", "is not a registered member")],
               "number": ("days since purchase", "within {n} days of purchase", (7, 60)),
               "prohibitions": [("the item was bought on final sale", "the item was bought at full price")]},
    "leave": {"action": "approve the leave request", "subject": "employee",
              "conditions": [("submitted the request in writing", "asked only verbally"),
                             ("has manager sign-off", "has no manager sign-off"),
                             ("finished the probation period", "is still on probation")],
              "number": ("months of service", "at least {n} months of service", (3, 24)),
              "prohibitions": [("the dates fall in the year-end freeze", "the dates are outside any freeze period")]},
    "loan": {"action": "approve the loan", "subject": "applicant",
             "conditions": [("provided proof of income", "provided no proof of income"),
                            ("has a guarantor", "has no guarantor"),
                            ("passed the identity check", "failed the identity check")],
             "number": ("credit score", "a credit score of at least {n}", (600, 750)),
             "prohibitions": [("the applicant has an unpaid default", "the applicant has no defaults")]},
    "access": {"action": "grant access to the server room", "subject": "contractor",
               "conditions": [("completed safety training", "has not completed safety training"),
                              ("is escorted by staff", "is not escorted"),
                              ("signed the visitor log", "did not sign the visitor log")],
               "number": ("hours since the badge was issued", "a badge issued within the last {n} hours", (4, 48)),
               "prohibitions": [("the contractor's badge was reported lost", "the badge has not been reported lost")]},
    "warranty": {"action": "accept the warranty claim", "subject": "owner",
                 "conditions": [("registered the product", "never registered the product"),
                                ("kept the serial number label intact", "removed the serial number label"),
                                ("used an authorised repair shop", "used an unauthorised repair shop")],
                 "number": ("months since purchase", "within {n} months of purchase", (6, 36)),
                 "prohibitions": [("the damage was caused by a drop", "there is no sign of physical damage")]},
}


def choose_number(rng, rule):
    lo, hi = rule[2]
    n = rng.randint(lo, hi)
    within = "within" in rule[1] and "credit" not in rule[1]
    # "within n" means value <= n; "at least n" means value >= n
    ok = rng.random() < 0.5
    if within:
        value = rng.randint(max(0, n - (n // 2 or 1)), n) if ok else rng.randint(n + 1, n + max(2, n // 2))
    else:
        value = rng.randint(n, n + max(2, n // 3)) if ok else rng.randint(max(0, n - max(2, n // 3)), n - 1)
    return n, value, ok


def eligibility(rng, topic_name):
    t = TOPICS[topic_name]
    conds = rng.sample(t["conditions"], rng.randint(1, 3))
    n, value, num_ok = choose_number(rng, t["number"])
    prohibition = t["prohibitions"][0]
    rules = [c[0] for c in conds] + [t["number"][1].format(n=n)]
    policy = f"Policy: to {t['action']}, the {t['subject']} must meet all of these: " + "; ".join(rules) + "."
    policy += f" Even then, do not {t['action']} if {prohibition[0]}."
    # Pick the answer first, then build facts that produce it, so no wording or count of facts predicts it
    permitted = rng.random() < 0.5
    if permitted:
        cond_state = ["good"] * len(conds)
        blocked = False
        if not num_ok:
            n, value, num_ok = choose_number(rng, t["number"])
            while not num_ok:
                n, value, num_ok = choose_number(rng, t["number"])
            rules[-1] = t["number"][1].format(n=n)
            policy = (f"Policy: to {t['action']}, the {t['subject']} must meet all of these: " + "; ".join(rules)
                      + f". Even then, do not {t['action']} if {prohibition[0]}.")
    else:
        failures = rng.sample(["bad", "unstated", "number", "blocked"], rng.randint(1, 2))
        cond_state = ["good"] * len(conds)
        k = rng.randrange(len(conds))
        if "bad" in failures:
            cond_state[k] = "bad"
        elif "unstated" in failures:
            cond_state[k] = "unstated"  # not stated: an unproven requirement counts as not met
        blocked = "blocked" in failures
        if "number" in failures:
            while num_ok:
                n, value, num_ok = choose_number(rng, t["number"])
            rules[-1] = t["number"][1].format(n=n)
            policy = (f"Policy: to {t['action']}, the {t['subject']} must meet all of these: " + "; ".join(rules)
                      + f". Even then, do not {t['action']} if {prohibition[0]}.")
        elif not num_ok:
            pass  # a failing number is one more reason to refuse
    facts = []
    for (good, bad), s in zip(conds, cond_state):
        if s == "good":
            facts.append(f"The {t['subject']} {good}.")
        elif s == "bad":
            facts.append(f"The {t['subject']} {bad}.")
    facts.append(f"Recorded {t['number'][0]}: {value}.")
    facts.append(f"Note: {prohibition[0] if blocked else prohibition[1]}.")
    rng.shuffle(facts)
    # The answer is recomputed from the final facts, so it is always consistent with the text
    assert permitted == (all(s == "good" for s in cond_state) and num_ok and not blocked)
    negated = rng.random() < 0.5
    if negated:
        question = f"Should the request be refused, so that we do not {t['action']}?"
        label = not permitted
    else:
        question = f"Under the policy, should we {t['action']}? Treat any requirement that is not stated as not met."
        label = permitted
    return policy + "\n" + " ".join(facts), question, {"type": "noul", "instructions": question, "label": label}


def payout(rng):
    amount = rng.randrange(1000, 40000, 100)
    deductible = rng.choice([250, 500, 1000, 2500])
    limit = rng.randrange(5000, 30000, 1000)
    excluded = rng.random() < 0.25
    cause = rng.choice(["a burst pipe", "a kitchen fire", "a storm", "theft"])
    exclusion = rng.choice(["gradual seepage", "flood water", "wear and tear"])
    claim_cause = exclusion if excluded else cause
    state = (f"Policy terms: losses are paid after a deductible of ${deductible:,}, up to a limit of ${limit:,} per claim. "
             f"Losses caused by {exclusion} are excluded. "
             f"Claim: the insured reports damage of ${amount:,} caused by {claim_cause}.")
    due = max(0, min(amount - deductible, limit))
    candidates = {
        "deny": "Deny the claim: the cause is excluded",
        "pay_capped": f"Pay ${due:,}",
        "pay_uncapped": f"Pay ${max(0, amount - deductible):,}",
        "pay_full": f"Pay ${amount:,}",
        "pay_limit": f"Pay ${limit:,}",
    }
    gold = "deny" if excluded else "pay_capped"
    # Drop options whose text duplicates the gold text, keep the gold plus up to three distinct distractors
    seen, keys = {candidates[gold]}, [gold]
    for k in rng.sample(list(candidates), len(candidates)):
        if k != gold and candidates[k] not in seen and len(keys) < 4:
            keys.append(k)
            seen.add(candidates[k])
    rng.shuffle(keys)
    criteria = {k: candidates[k] for k in keys}
    question = "Under the policy terms, what should be paid on this claim?"
    return state, question, {"type": "choice", "instructions": question, "criteria": criteria, "label": gold}


CHECKS = [
    ("Every shipment must have a tracking number.", "Shipment {i} has no tracking number.", "Shipment {i} has a tracking number."),
    ("Hazardous items must be labelled.", "A hazardous item in shipment {i} is unlabelled.", "All hazardous items in shipment {i} are labelled."),
    ("Weight must not exceed 30 kg per box.", "A box in shipment {i} weighs {w} kg.", "The heaviest box in shipment {i} weighs {v} kg."),
    ("Customs forms are needed for international shipments.", "Shipment {i} is international and has no customs form.", "Shipment {i} is domestic."),
    ("Signature is required for items over $500.", "Shipment {i} holds a $900 item sent without signature.", "Shipment {i} holds only items under $500."),
]


def violations(rng):
    picked = rng.sample(CHECKS, rng.randint(3, 5))
    target = rng.choice([0, 1, 2, 3, 3, 4])  # make "three or more" reachable
    target = min(target, len(picked))
    bad = set(rng.sample(range(len(picked)), target))
    i = rng.randint(10, 99)
    facts = []
    for k, (_, broken, fine) in enumerate(picked):
        facts.append((broken if k in bad else fine).format(i=i, w=rng.randint(31, 45), v=rng.randint(5, 30)))
    rng.shuffle(facts)
    state = "Rules: " + " ".join(r for r, _, _ in picked) + "\nAudit notes: " + " ".join(facts)
    levels = ["No violations", "Exactly one violation", "Exactly two violations", "Three or more violations"]
    question = f"How many of the rules does shipment {i} violate?"
    return state, question, {"type": "score", "instructions": question, "criteria": levels, "label": min(len(bad), 3)}


def to_json_state(rng, state):
    """Sometimes present the same facts as a JSON object with a text field, the way structured inputs arrive."""
    if rng.random() < 0.25:
        return json.dumps({"case_id": f"C-{rng.randint(1000, 9999)}", "details": state}, ensure_ascii=False)
    return state


def generate(n, seed=0):
    rng = random.Random(seed)
    rows = []
    makers = ["eligibility"] * 5 + ["payout"] * 3 + ["violations"] * 2
    for _ in range(n):
        kind = rng.choice(makers)
        if kind == "eligibility":
            state, _, q = eligibility(rng, rng.choice(list(TOPICS)))
        elif kind == "payout":
            state, _, q = payout(rng)
        else:
            state, _, q = violations(rng)
        rows.append({"state": to_json_state(rng, state), "questions_json": json.dumps({"q": q}),
                     "domain": f"rules_{kind}", "question_types": q["type"], "question_count": 1})
    return rows
