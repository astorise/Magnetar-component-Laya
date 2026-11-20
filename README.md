# magnetar-component-laya

Placeholder for Laya decision models in [Magnetar](https://github.com/astorise/Magnetar).

## Status

This is a scaffold only. It has no model graph, loader, weights, inference implementation, Component Model exports or supported checkpoint. Creating this repository does not make the model runnable in Magnetar.

## Work required

A compatible ModernBERT or mmBERT encoder, the trained decision head, option scoring and structured decisions.

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/convaiinnovations/laya). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/laya`.
