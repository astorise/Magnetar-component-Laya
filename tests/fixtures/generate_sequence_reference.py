"""Regenerate sequence references from a pinned local upstream rl_common.py."""

import ast
import hashlib
import json
import sys
import typing
from pathlib import Path

source = Path(sys.argv[1]).read_bytes()
expected = "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e"
if hashlib.sha256(source).hexdigest() != expected:
    raise ValueError("rl_common.py does not match the pinned upstream revision")
tree = ast.parse(source)
names = {"serialize_state", "render_options", "build_sequence"}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
namespace = {"json": json, "Dict": typing.Dict, "List": typing.List, "Optional": typing.Optional}
exec(compile(ast.Module(body=functions, type_ignores=[]), "rl_common.py", "exec"), namespace)


class ToyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 103
    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(character) + 1000 for character in text]}


cases = [
    {"type": "choice", "instructions": "Choose [MASK] a department", "criteria": {"billing": "billing queries", "support": None, "sales": "buy"}, "state": "hello [MASK] world", "max_len": 128, "head_max_len": 64, "truncate_left": False},
    {"type": "score", "instructions": "Rate urgency", "criteria": ["low", "middle", "high", "emergency"], "state": "xyz" * 40, "max_len": 80, "head_max_len": 24, "truncate_left": False},
    {"type": "noul", "instructions": "Is this urgent?", "criteria": None, "state": "abcdefghijklmnopqrstuvwxyz" * 3, "max_len": 60, "head_max_len": 20, "truncate_left": True},
    {"type": "choice", "instructions": "x", "criteria": {"a": None}, "state": "abcdef", "max_len": 0, "head_max_len": 24, "truncate_left": True},
]
tokenizer = ToyTokenizer()
for case in cases:
    question = {"t": case["type"], "ins": case["instructions"], "crit": case["criteria"]}
    if case["max_len"] == 0:
        base, _ = namespace["build_sequence"](tokenizer, "", question, 1000, case["head_max_len"], truncate_left=True)
        case["max_len"] = len(base)
    ids, markers = namespace["build_sequence"](tokenizer, case["state"], question, case["max_len"], case["head_max_len"], truncate_left=case["truncate_left"])
    assert len(markers) == len(namespace["render_options"](question))
    case["input_ids"] = ids
    case["marker_positions"] = markers
    if case["type"] == "choice":
        case["choices_ordered"] = [[label, description] for label, description in case["criteria"].items()]

(Path(__file__).resolve().parent / "sequence_reference.json").write_text(
    json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
