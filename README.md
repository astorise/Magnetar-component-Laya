# magnetar-component-laya

Laya decision component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library prepares typed question sequences using a caller-provided
tokenizer, evaluates the learned decision head from supplied f32 encoder states
and weights, and post-processes option logits into choice, score and noul
outputs with the checkpoint's temperatures. The CPU reference includes the
type embedding, two-stage Transformer head, option scorer and action head.
Synthetic padded and variable-option tests compare its raw scores and
probabilities to the pinned upstream PyTorch implementation, including a
multi-head case. This does not establish calibration of a deployment.
Encoder execution, checkpoint loading, graph production and Component Model
exports remain unimplemented; Laya is not yet runnable in Magnetar.

## Work required

A compatible ModernBERT or mmBERT encoder, loading the trained decision-head
weights and a runtime decision interface remain necessary. Magnetar's missing
encoder and decision contracts are tracked in [issue #214](https://github.com/astorise/Magnetar/issues/214).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/convaiinnovations/laya). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/laya`.
