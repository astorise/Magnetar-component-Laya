# Laya post-processing fixtures

`rl_agent_config.json` is the unmodified checkpoint configuration from
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya/blob/fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1/rl_agent_config.json)
at revision `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1` (SHA-256
`ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd`).

`postprocessing_reference.json` contains independent Python standard-library
calculations for fixed synthetic logits, using the same temperature bucket,
softmax, typed output and entropy formulas as the checkpoint's
[`rl_agent_api.py`](https://huggingface.co/convaiinnovations/laya/blob/fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1/rl_agent_api.py)
and `rl_common.py`. It does not contain model outputs or weights. The upstream
model card declares Apache-2.0; this is not an inference accuracy benchmark.
Regenerate it with `python tests/fixtures/generate_reference.py`.

`sequence_reference.json` was produced by running the unmodified
`serialize_state`, `render_options` and `build_sequence` functions from the
same pinned `rl_common.py`, using a deterministic test tokenizer (each
character's Unicode value plus 1000, with CLS=101, SEP=102, MASK=103). It covers
choice, score, noul, marker text removal, head truncation and left-truncated
state. It is not a substitute for testing the published tokenizer.
To reproduce it, download the pinned `rl_common.py` linked above and run
`python tests/fixtures/generate_sequence_reference.py path/to/rl_common.py`;
the generator verifies the upstream file's SHA-256 before use.
