"""Run the pinned Laya DecisionModel with synthetic encoder outputs and weights.

Usage: python generate_head_reference.py /path/to/rl_common.py
Requires PyTorch. The encoder is a fixed-output test double; the decision head
itself is the unmodified class extracted from the pinned upstream source.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e"
tree = ast.parse(source)
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DecisionModel")
scope = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned_rl_common.py", "exec"), scope)


def patterned(name, shape):
    count = 1
    for dim in shape:
        count *= dim
    seed = sum((i + 1) * byte for i, byte in enumerate(name.encode("ascii"))) % 97
    values = ((torch.arange(count, dtype=torch.int64) * 17 + seed) % 101 - 50).float() / 200.0
    if "norm" in name and name.endswith(".weight"):
        values += 1.0
    return values.reshape(shape)


class Encoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.value = None

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.value)


config = json.loads(Path(__file__).with_name("rl_agent_config.json").read_text(encoding="utf-8"))
results = []
for hidden, tokens, mask, markers, qtype in [
    (8, 7, [1, 1, 1, 1, 1, 0, 0], [1, 3, 4], 0),
    (8, 6, [1, 1, 1, 1, 1, 1], [1, 4], 2),
    (128, 4, [1, 1, 1, 0], [1, 2], 1),
]:
    encoder = Encoder(hidden)
    model = scope["DecisionModel"](encoder, head_layers=2, n_act=2, dropout=0.0).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(patterned(name, parameter.shape))
        encoder.value = patterned("encoder_output", (1, tokens, hidden))
        logits, action = model(
            torch.zeros((1, tokens), dtype=torch.long),
            torch.tensor([mask]),
            torch.tensor([markers]),
            torch.ones((1, len(markers)), dtype=torch.bool),
            torch.tensor([qtype]),
        )
    kind = ["choice", "score", "noul"][qtype]
    bucket = "2" if len(markers) <= 2 else "3-5" if len(markers) <= 5 else "6-10" if len(markers) <= 10 else "11+"
    temperature = config["temperature_by_options"].get(f"{kind}:{bucket}", config["temperature"][qtype])
    probabilities = torch.softmax(logits.float() / temperature, -1).flatten().tolist()
    action_probability = torch.softmax(action.float(), -1)[0, 0].item()
    results.append({"hidden": hidden, "tokens": tokens, "mask": mask,
                    "markers": markers, "qtype": qtype,
                    "logits": logits.flatten().tolist(),
                    "action_logits": action.flatten().tolist(),
                    "probabilities": probabilities,
                    "action_probability": action_probability})

destination = Path(__file__).with_name("head_reference.json")
destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
