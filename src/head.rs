//! CPU reference for Laya's learned decision head, given encoder hidden states.
//!
//! This is an inference-only reference for the pinned `DecisionModel`: type
//! embedding, pre-norm TransformerEncoder layers, marker scorer and action
//! head. It does not load a checkpoint or execute the ModernBERT encoder.

use crate::{Calibrator, Decision, DecisionError, QuestionType, softmax as decision_softmax};
use std::{error::Error, fmt};

const MAX_ELEMENTS: usize = 67_108_864;

/// Row-major PyTorch linear weights `[output, input]` and bias `[output]`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Learned LayerNorm scale and bias, with upstream epsilon `1e-5`.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One evaluation-mode PyTorch TransformerEncoderLayer (`norm_first=true`).
#[derive(Clone, Debug)]
pub struct TransformerLayer {
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    /// Q, K and V stacked in that order: `[3 * hidden, hidden]`.
    pub qkv: Linear,
    pub attention_output: Linear,
    /// Feed-forward width is exactly four times the hidden width.
    pub feed_forward_in: Linear,
    pub feed_forward_out: Linear,
}

/// Checkpoint parameters for the head. Matrices use PyTorch's output-major
/// layout, and `type_embedding` has shape `[3, hidden]`.
#[derive(Clone, Debug)]
pub struct DecisionHead {
    pub hidden: usize,
    pub type_embedding: Vec<f32>,
    pub layers: Vec<TransformerLayer>,
    pub scorer_norm: LayerNorm,
    pub scorer_in: Linear,
    pub scorer_out: Linear,
    /// Action head is `[hidden + 4, 256, action_count]`.
    pub action_in: Linear,
    pub action_out: Linear,
}

/// Raw scores before the checkpoint's temperature and answer conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadOutput {
    pub option_logits: Vec<f32>,
    pub action_logits: Vec<f32>,
}

/// Typed result retaining raw option and action scores alongside probabilities.
/// The first action is "act" in the pinned model; later actions follow the
/// checkpoint configuration's action order. No action threshold is implied.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResult {
    pub option_scores: Vec<f32>,
    pub decision: Decision,
    pub action_scores: Vec<f32>,
    pub action_probabilities: Vec<f64>,
}

impl HeadOutput {
    /// Apply the checkpoint's question-type temperature and action softmax.
    /// Temperature adjustment alone does not establish statistical calibration.
    pub fn decide(
        &self,
        calibrator: &Calibrator,
        question_type: QuestionType,
    ) -> Result<DecisionResult, DecisionError> {
        let option_scores: Vec<_> = self
            .option_logits
            .iter()
            .map(|&value| value as f64)
            .collect();
        let action_scores: Vec<_> = self
            .action_logits
            .iter()
            .map(|&value| value as f64)
            .collect();
        if action_scores.len() < 2 {
            return Err(DecisionError(
                "the action head requires at least two scores".into(),
            ));
        }
        Ok(DecisionResult {
            option_scores: self.option_logits.clone(),
            decision: calibrator.decide(question_type, &option_scores)?,
            action_scores: self.action_logits.clone(),
            action_probabilities: decision_softmax(&action_scores, 1.0)?,
        })
    }
}

/// A malformed tensor, invalid mask or non-finite intermediate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadError(pub String);

impl fmt::Display for HeadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for HeadError {}

fn product(a: usize, b: usize) -> Result<usize, HeadError> {
    a.checked_mul(b)
        .filter(|&n| n <= MAX_ELEMENTS)
        .ok_or_else(|| HeadError("decision head dimensions exceed reference limits".into()))
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

impl Linear {
    fn validate(&self, input: usize, output: usize) -> Result<(), HeadError> {
        if self.weight.len() != product(input, output)?
            || self.bias.len() != output
            || !finite(&self.weight)
            || !finite(&self.bias)
        {
            return Err(HeadError("invalid decision-head linear parameters".into()));
        }
        Ok(())
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        self.weight
            .chunks_exact(input.len())
            .zip(&self.bias)
            .map(|(row, &bias)| {
                row.iter()
                    .zip(input)
                    .fold(bias, |sum, (&weight, &value)| sum + weight * value)
            })
            .collect()
    }
}

impl LayerNorm {
    fn validate(&self, hidden: usize) -> Result<(), HeadError> {
        if self.weight.len() != hidden
            || self.bias.len() != hidden
            || !finite(&self.weight)
            || !finite(&self.bias)
        {
            return Err(HeadError(
                "invalid decision-head LayerNorm parameters".into(),
            ));
        }
        Ok(())
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, HeadError> {
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / input.len() as f32;
        if !mean.is_finite() || !variance.is_finite() {
            return Err(HeadError("non-finite decision-head LayerNorm".into()));
        }
        let adjusted = variance + 1e-5;
        if !adjusted.is_finite() {
            return Err(HeadError("non-finite decision-head variance".into()));
        }
        let inverse = adjusted.sqrt().recip();
        let result: Vec<_> = input
            .iter()
            .zip(&self.weight)
            .zip(&self.bias)
            .map(|((&value, &weight), &bias)| (value - mean) * inverse * weight + bias)
            .collect();
        if !finite(&result) {
            return Err(HeadError("non-finite decision-head normalization".into()));
        }
        Ok(result)
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn softmax(values: &[f32]) -> Result<Vec<f32>, HeadError> {
    if values.is_empty() || !finite(values) {
        return Err(HeadError(
            "decision-head softmax requires finite values".into(),
        ));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<_> = values.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = result.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(HeadError("invalid decision-head softmax".into()));
    }
    for value in &mut result {
        *value /= sum;
    }
    Ok(result)
}

impl DecisionHead {
    fn validate(&self) -> Result<usize, HeadError> {
        let d = self.hidden;
        let heads = (d / 64).max(1);
        if d == 0
            || d > 4096
            || !d.is_multiple_of(heads)
            || self.layers.len() > 16
            || self.type_embedding.len() != product(3, d)?
            || !finite(&self.type_embedding)
        {
            return Err(HeadError(
                "invalid decision-head geometry or type embedding".into(),
            ));
        }
        let expanded = product(4, d)?;
        for layer in &self.layers {
            layer.norm1.validate(d)?;
            layer.norm2.validate(d)?;
            layer.qkv.validate(d, product(3, d)?)?;
            layer.attention_output.validate(d, d)?;
            layer.feed_forward_in.validate(d, expanded)?;
            layer.feed_forward_out.validate(expanded, d)?;
        }
        self.scorer_norm.validate(d)?;
        self.scorer_in.validate(d, d)?;
        self.scorer_out.validate(d, 1)?;
        self.action_in.validate(d + 4, 256)?;
        if !(2..=16).contains(&self.action_out.bias.len()) {
            return Err(HeadError("invalid decision-head action count".into()));
        }
        self.action_out.validate(256, self.action_out.bias.len())?;
        Ok(heads)
    }

    /// Evaluate one sequence. `hidden_states` is `[tokens, hidden]`, `mask`
    /// selects valid attention keys and each marker selects a valid token.
    /// The upstream action head requires at least two option markers.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        tokens: usize,
        mask: &[bool],
        markers: &[usize],
        question_type: QuestionType,
    ) -> Result<HeadOutput, HeadError> {
        let heads = self.validate()?;
        let d = self.hidden;
        if tokens == 0
            || tokens > 4096
            || hidden_states.len() != product(tokens, d)?
            || !finite(hidden_states)
            || mask.len() != tokens
            || !mask.contains(&true)
            || markers.len() < 2
            || markers.len() > 255
            || markers
                .iter()
                .any(|&position| position >= tokens || !mask[position])
        {
            return Err(HeadError(
                "invalid decision-head input, mask or markers".into(),
            ));
        }
        product(tokens, product(4, d)?)?;
        let type_index = match question_type {
            QuestionType::Choice => 0,
            QuestionType::Score => 1,
            QuestionType::Noul => 2,
        };
        let mut states = hidden_states.to_vec();
        for row in states.chunks_exact_mut(d) {
            for (value, &embedding) in row
                .iter_mut()
                .zip(&self.type_embedding[type_index * d..(type_index + 1) * d])
            {
                *value += embedding;
            }
        }
        if !finite(&states) {
            return Err(HeadError("non-finite decision-head embedding".into()));
        }
        let width = d / heads;
        for layer in &self.layers {
            let mut qkv = Vec::with_capacity(tokens * 3 * d);
            for row in states.chunks_exact(d) {
                qkv.extend(layer.qkv.apply(&layer.norm1.apply(row)?));
            }
            if !finite(&qkv) {
                return Err(HeadError("non-finite decision-head projection".into()));
            }
            let mut attention = vec![0.0; tokens * d];
            for query in 0..tokens {
                for head in 0..heads {
                    let offset = head * width;
                    let mut keys = Vec::with_capacity(tokens);
                    let mut scores = Vec::with_capacity(tokens);
                    for (key, &valid) in mask.iter().enumerate() {
                        if !valid {
                            continue;
                        }
                        let mut dot = 0.0;
                        for dimension in 0..width {
                            dot += qkv[query * 3 * d + offset + dimension]
                                * qkv[key * 3 * d + d + offset + dimension];
                        }
                        keys.push(key);
                        scores.push(dot / (width as f32).sqrt());
                    }
                    let probabilities = softmax(&scores)?;
                    for dimension in 0..width {
                        let mut value = 0.0;
                        for (&key, &probability) in keys.iter().zip(&probabilities) {
                            value += probability * qkv[key * 3 * d + 2 * d + offset + dimension];
                        }
                        attention[query * d + offset + dimension] = value;
                    }
                }
            }
            for (row, attended) in states.chunks_exact_mut(d).zip(attention.chunks_exact(d)) {
                let projected = layer.attention_output.apply(attended);
                for (value, residual) in row.iter_mut().zip(projected) {
                    *value += residual;
                }
                let normalized = layer.norm2.apply(row)?;
                let intermediate = layer.feed_forward_in.apply(&normalized);
                if !finite(&intermediate) {
                    return Err(HeadError("non-finite decision-head feed-forward".into()));
                }
                let intermediate: Vec<_> = intermediate
                    .into_iter()
                    .map(|value| value.max(0.0))
                    .collect();
                let projected = layer.feed_forward_out.apply(&intermediate);
                for (value, residual) in row.iter_mut().zip(projected) {
                    *value += residual;
                }
            }
            if !finite(&states) {
                return Err(HeadError("non-finite decision-head layer".into()));
            }
        }
        let mut logits = Vec::with_capacity(markers.len());
        for &position in markers {
            let normalized = self
                .scorer_norm
                .apply(&states[position * d..(position + 1) * d])?;
            let hidden: Vec<_> = self
                .scorer_in
                .apply(&normalized)
                .into_iter()
                .map(gelu)
                .collect();
            logits.push(self.scorer_out.apply(&hidden)[0]);
        }
        let probabilities = softmax(&logits)?;
        let mut ranked = probabilities.clone();
        ranked.sort_by(|a, b| b.total_cmp(a));
        let entropy = -probabilities
            .iter()
            .map(|&p| p * p.max(1e-9).ln())
            .sum::<f32>()
            / (markers.len().max(2) as f32).ln();
        let mut action_input = states[..d].to_vec();
        action_input.extend([
            ranked[0],
            ranked[0] - ranked[1],
            entropy,
            markers.len() as f32 / 255.0,
        ]);
        let action_hidden: Vec<_> = self
            .action_in
            .apply(&action_input)
            .into_iter()
            .map(gelu)
            .collect();
        let action_logits = self.action_out.apply(&action_hidden);
        if !finite(&action_logits) {
            return Err(HeadError("non-finite decision-head action".into()));
        }
        Ok(HeadOutput {
            option_logits: logits,
            action_logits,
        })
    }
}
