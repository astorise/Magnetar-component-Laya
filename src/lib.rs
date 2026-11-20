//! Post-processing for the Laya decision model.
//!
//! This implements the upstream temperature selection and conversion from
//! decision-head logits to distributions. It does not run the encoder or
//! decision head: callers must supply their actual logits.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// A question type supported by the Laya checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// A typed prediction derived from the option distribution.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    /// Index of the option with the highest probability; ties choose the first.
    Choice(usize),
    /// Probability-weighted mean of the zero-based option indices.
    Score(f64),
    /// Probability of the `true` option (the second of exactly two options).
    Noul(f64),
}

/// Laya's output before API-specific rounding or option-label rendering.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub answer: Answer,
    pub probabilities: Vec<f64>,
    /// One minus normalized entropy for choice and score; absent for noul.
    /// This is a confidence statistic, not proof of calibration.
    pub confidence: Option<f64>,
}

/// Post-training temperature settings from a Laya `rl_agent_config.json`.
#[derive(Debug, Clone)]
pub struct Calibrator {
    temperatures: [f64; 3],
    by_option_count: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct RawConfig {
    #[serde(default = "default_temperatures")]
    temperature: [f64; 3],
    #[serde(default)]
    temperature_by_options: BTreeMap<String, f64>,
}

fn default_temperatures() -> [f64; 3] {
    [1.0; 3]
}

impl Calibrator {
    /// Load the checkpoint's temperature settings. Missing settings default
    /// to unit temperatures, as in the upstream inference implementation.
    pub fn from_config(bytes: &[u8]) -> Result<Self, DecisionError> {
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| DecisionError(format!("invalid Laya config: {error}")))?;
        if raw
            .temperature
            .iter()
            .chain(raw.temperature_by_options.values())
            .any(|temperature| !temperature.is_finite() || *temperature <= 0.0)
        {
            return Err(DecisionError(
                "temperatures must be finite and positive".into(),
            ));
        }
        Ok(Self {
            temperatures: raw.temperature,
            by_option_count: raw.temperature_by_options,
        })
    }

    /// Select the upstream question-type and option-count temperature.
    pub fn temperature(&self, kind: QuestionType, options: usize) -> f64 {
        let bucket = match options {
            0..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        self.by_option_count
            .get(&format!("{}:{bucket}", kind.name()))
            .copied()
            .unwrap_or(self.temperatures[kind.index()])
    }

    /// Convert one question's unmasked option logits into a typed decision.
    ///
    /// `logits` must contain exactly the real options, without padded
    /// markers. The model must supply both noul logits in false/true order.
    pub fn decide(&self, kind: QuestionType, logits: &[f64]) -> Result<Decision, DecisionError> {
        if logits.is_empty() {
            return Err(DecisionError(
                "a question must have at least one option".into(),
            ));
        }
        if kind == QuestionType::Noul && logits.len() != 2 {
            return Err(DecisionError("noul requires false and true logits".into()));
        }
        let probabilities = softmax(logits, self.temperature(kind, logits.len()))?;
        let answer = match kind {
            QuestionType::Choice => {
                let mut best = 0;
                for index in 1..probabilities.len() {
                    if probabilities[index] > probabilities[best] {
                        best = index;
                    }
                }
                Answer::Choice(best)
            }
            QuestionType::Score => Answer::Score(
                probabilities
                    .iter()
                    .enumerate()
                    .map(|(index, probability)| index as f64 * probability)
                    .sum(),
            ),
            QuestionType::Noul => Answer::Noul(probabilities[1]),
        };
        let confidence = (kind != QuestionType::Noul).then(|| entropy_confidence(&probabilities));
        Ok(Decision {
            answer,
            probabilities,
            confidence,
        })
    }
}

/// Softmax of the two action-head logits, returning the `act` probability.
/// This is separate from the calibrated answer distribution.
pub fn act_probability(logits: [f64; 2]) -> Result<f64, DecisionError> {
    Ok(softmax(&logits, 1.0)?[0])
}

fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>, DecisionError> {
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(DecisionError("logits must be finite".into()));
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<_> = logits
        .iter()
        .map(|value| ((value - maximum) / temperature).exp())
        .collect();
    let total: f64 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

fn entropy_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy = -probabilities
        .iter()
        .map(|p| p * p.clamp(1e-12, 1.0).ln())
        .sum::<f64>();
    1.0 - entropy / (probabilities.len() as f64).ln()
}

/// Invalid configuration or inputs to Laya decision post-processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionError(String);

impl fmt::Display for DecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for DecisionError {}
