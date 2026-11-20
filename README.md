# magnetar-component-laya

Laya decision component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library prepares typed question sequences using a caller-provided
tokenizer and implements checkpoint temperature selection and post-processing
of supplied decision-head logits into choice, score and noul outputs.
Integration tests compare against independently calculated outputs
using the published checkpoint configuration. These functions do not run the
encoder or learned head and do not establish calibration of a deployment.
Model loading, graph production, Component Model exports and inference are not
implemented; Laya is not yet runnable in Magnetar.

## Work required

A compatible ModernBERT or mmBERT encoder, the trained decision head, option
scoring and a runtime decision interface remain necessary. Magnetar's missing
encoder and decision contracts are tracked in [issue #214](https://github.com/astorise/Magnetar/issues/214).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/convaiinnovations/laya). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/laya`.
