"""Run the pinned Laya head on synthetic encoder outputs and real F16 weights.

The header and two byte ranges are read from the pinned model.safetensors;
their exact SHA-256 hashes are checked. The output directory contains decoded
f32 weights for the ignored Rust integration test and must not be committed.
Requires PyTorch and NumPy. No encoder checkpoint is loaded.
"""

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


HASHES = {
    "source": "8d83611d480c971d640a7b7d3aa2f2219c5e8455e9cc2329fd073681bd8be23e",
    "header": "3a42d8e7a96d5aa22c5bc9475d7ff092e32a832ae7c6973688f66f7a84048c04",
    "start": "cacecb11b58bbdfac01af68eb042f063036da932bc00bfcec3cbfca531dbaad6",
    "end": "5450d8a578601e6a6fb0089b66d2e95cc7d0904a278c21e5f4bc85f56e6538d7",
}


def verify(data, name):
    digest = hashlib.sha256(data).hexdigest()
    if digest != HASHES[name]:
        raise ValueError(f"unexpected {name} SHA-256: {digest}")
    return data


def checked(path, name):
    return verify(path.read_bytes(), name)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1024)
        self.value = None

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.value)


def patterned(count):
    seed = sum((index + 1) * byte for index, byte in enumerate(b"encoder_output")) % 97
    return (((torch.arange(count, dtype=torch.int64) * 17 + seed) % 101 - 50).float() / 200.0)


def main():
    if sys.byteorder != "little":
        raise RuntimeError("checkpoint decoding requires a little-endian host")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "out", "reference"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ["model", "header", "start", "end"]:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    source = checked(args.source, "source")
    if args.model is not None:
        if any((args.header, args.start, args.end)):
            parser.error("use --model or the three separate ranges, not both")
        with args.model.open("rb") as model_file:
            length = int.from_bytes(model_file.read(8), "little")
            if length != 21536:
                raise ValueError("unexpected checkpoint header length")
            header_bytes = verify(model_file.read(length), "header")
            start = verify(model_file.read(527888), "start")
            model_file.seek(8 + length + 790091280)
            end = verify(model_file.read(52496386), "end")
    else:
        if not all((args.header, args.start, args.end)):
            parser.error("provide --model or all of --header, --start and --end")
        header_bytes = checked(args.header, "header")
        start, end = checked(args.start, "start"), checked(args.end, "end")
    header = json.loads(header_bytes)
    ranges = [(0, start), (790091280, end)]
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DecisionModel")
    scope = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "pinned_rl_common.py", "exec"), scope)
    torch.set_num_threads(1)
    encoder = Encoder()
    model = scope["DecisionModel"](encoder, head_layers=2, n_act=2, dropout=0.0).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    state = {}
    for name, descriptor in header.items():
        if name == "__metadata__" or name.startswith("encoder."):
            continue
        start, end = descriptor["data_offsets"]
        region = next((data[start - base : end - base] for base, data in ranges if base <= start and end <= base + len(data)), None)
        if region is None:
            raise ValueError(f"missing checkpoint bytes for {name}")
        expected_dtype = "F32" if name == "temperature" else "F16"
        if descriptor["dtype"] != expected_dtype:
            raise ValueError(f"unexpected checkpoint dtype for {name}")
        dtype = torch.float32 if expected_dtype == "F32" else torch.float16
        value = torch.frombuffer(bytearray(region), dtype=dtype).reshape(descriptor["shape"]).float()
        state[name] = value
        if name != "temperature":
            (args.out / f"{name}.f32").write_bytes(value.numpy().astype("<f4", copy=False).tobytes())
    model.load_state_dict(state, strict=True)
    config = json.loads(Path(__file__).with_name("rl_agent_config.json").read_text(encoding="utf-8"))
    cases = []
    for mask, markers, qtype in [([1, 1, 1, 0], [1, 2], 0), ([1, 1, 1, 1], [1, 2, 3], 1)]:
        encoder.value = patterned(4 * 1024).reshape(1, 4, 1024)
        with torch.inference_mode():
            logits, actions = model(
                torch.zeros((1, 4), dtype=torch.long), torch.tensor([mask]),
                torch.tensor([markers]), torch.ones((1, len(markers)), dtype=torch.bool),
                torch.tensor([qtype]),
            )
        kind = ["choice", "score", "noul"][qtype]
        bucket = "2" if len(markers) <= 2 else "3-5"
        temperature = config["temperature_by_options"].get(
            f"{kind}:{bucket}", config["temperature"][qtype]
        )
        probabilities = torch.softmax(logits.float() / temperature, -1).flatten()
        answer = probabilities.argmax().item() if qtype == 0 else sum(
            index * probability for index, probability in enumerate(probabilities.tolist())
        )
        cases.append({
            "mask": mask, "markers": markers, "qtype": qtype,
            "logits": logits.flatten().tolist(), "action_logits": actions.flatten().tolist(),
            "probabilities": probabilities.tolist(), "answer": answer,
            "action_probabilities": torch.softmax(actions.float(), -1).flatten().tolist(),
        })
    args.reference.write_text(json.dumps({"source_hashes": HASHES, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(state) - 1} decoded head tensors and {len(cases)} reference cases")


if __name__ == "__main__":
    main()
