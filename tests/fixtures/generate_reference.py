"""Regenerate synthetic-logit references using the pinned upstream formulas."""

import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
config = json.loads((root / "rl_agent_config.json").read_text(encoding="utf-8"))
cases = [
    ("choice", [0.0, 1.0, -2.0]),
    ("choice", [3.0, -1.0]),
    ("choice", [float(i) - 4 for i in range(11)]),
    ("score", [0.5, -0.25, 1.75, 0.0]),
    ("noul", [1.25, -0.75]),
    ("score", [7.0]),
]
results = []
for kind, logits in cases:
    count = len(logits)
    bucket = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    kind_index = {"choice": 0, "score": 1, "noul": 2}[kind]
    temperature = config["temperature_by_options"].get(
        f"{kind}:{bucket}", config["temperature"][kind_index]
    )
    scaled = [value / temperature for value in logits]
    probabilities = [math.exp(value - max(scaled)) for value in scaled]
    total = sum(probabilities)
    probabilities = [value / total for value in probabilities]
    confidence = (
        1.0
        if count < 2
        else 1 + sum(p * math.log(max(p, 1e-12)) for p in probabilities) / math.log(count)
    )
    if kind == "choice":
        answer = max(range(count), key=probabilities.__getitem__)
    elif kind == "score":
        answer = sum(index * p for index, p in enumerate(probabilities))
    else:
        answer = probabilities[1]
    results.append(
        {
            "kind": kind,
            "logits": logits,
            "temperature": temperature,
            "probabilities": probabilities,
            "answer": answer,
            "confidence": None if kind == "noul" else confidence,
        }
    )

(root / "postprocessing_reference.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8"
)
