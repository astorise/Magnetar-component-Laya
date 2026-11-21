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

/// A choice and its optional explanatory criterion, in presentation order.
#[derive(Clone, Copy, Debug)]
pub struct ChoiceOption<'a> {
    pub label: &'a str,
    pub description: Option<&'a str>,
}

/// A typed question in the upstream inference format.
#[derive(Clone, Debug)]
pub enum Question<'a> {
    Choice {
        instructions: &'a str,
        options: &'a [ChoiceOption<'a>],
    },
    Score {
        instructions: &'a str,
        levels: &'a [&'a str],
    },
    Noul {
        instructions: &'a str,
        when_false: Option<&'a str>,
        when_true: Option<&'a str>,
    },
}

impl Question<'_> {
    fn render(&self) -> (QuestionType, &str, Vec<String>) {
        match self {
            Self::Choice {
                instructions,
                options,
            } => (
                QuestionType::Choice,
                instructions,
                options
                    .iter()
                    .map(|option| match option.description {
                        Some(description) if !description.is_empty() => {
                            format!("{}: {description}", option.label)
                        }
                        _ => option.label.to_string(),
                    })
                    .collect(),
            ),
            Self::Score {
                instructions,
                levels,
            } => (
                QuestionType::Score,
                instructions,
                levels
                    .iter()
                    .enumerate()
                    .map(|(i, level)| format!("level {i}: {level}"))
                    .collect(),
            ),
            Self::Noul {
                instructions,
                when_false,
                when_true,
            } => (
                QuestionType::Noul,
                instructions,
                vec![
                    format!(
                        "false: {}",
                        when_false
                            .filter(|s| !s.is_empty())
                            .unwrap_or("no, the statement does not hold")
                    ),
                    format!(
                        "true: {}",
                        when_true
                            .filter(|s| !s.is_empty())
                            .unwrap_or("yes, the statement holds")
                    ),
                ],
            ),
        }
    }
}

/// Special-token IDs and the literal mask token of the selected tokenizer.
#[derive(Clone, Copy, Debug)]
pub struct SpecialTokens<'a> {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub mask_text: &'a str,
}

/// Token sequence and positions read by the learned decision head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedSequence {
    pub input_ids: Vec<u32>,
    pub marker_positions: Vec<usize>,
    pub question_type: QuestionType,
}

/// Render and tokenize one question using Laya's pinned sequence layout.
///
/// `encode` must tokenize the text without automatically adding special
/// tokens, using the checkpoint's own tokenizer. State is passed as text;
/// callers must serialize non-string state before invoking this function.
/// All options must retain a marker after truncation or the call fails.
pub fn prepare_sequence<F, E>(
    question: &Question<'_>,
    state: &str,
    special: SpecialTokens<'_>,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
    encode: F,
) -> Result<PreparedSequence, DecisionError>
where
    F: Fn(&str) -> Result<Vec<u32>, E>,
    E: fmt::Display,
{
    if special.mask_text.is_empty() {
        return Err(DecisionError("mask token text must not be empty".into()));
    }
    if max_len == 0 {
        return Err(DecisionError("max_len must be positive".into()));
    }
    let (question_type, instructions, options) = question.render();
    if options.is_empty() {
        return Err(DecisionError(
            "a question must have at least one option".into(),
        ));
    }
    let tokenize = |text: &str| {
        encode(text).map_err(|error| DecisionError(format!("tokenization failed: {error}")))
    };
    let instructions = instructions.replace(special.mask_text, " ");
    let mut head = tokenize(&format!(
        "{} question: {instructions}",
        question_type.name()
    ))?;
    let mut option_tokens = options
        .iter()
        .map(|option| {
            let mut ids = vec![special.mask];
            let option = option.replace(special.mask_text, " ");
            ids.extend(tokenize(&format!(" {option}"))?.into_iter().take(48));
            Ok(ids)
        })
        .collect::<Result<Vec<_>, DecisionError>>()?;
    let mut option_size: usize = option_tokens.iter().map(Vec::len).sum();
    let mut budget = head_max_len.saturating_sub(option_size);
    if budget < 16 {
        let per_option = (head_max_len.saturating_sub(16) / option_tokens.len()).max(4);
        for ids in &mut option_tokens {
            ids.truncate(per_option);
        }
        option_size = option_tokens.iter().map(Vec::len).sum();
        budget = head_max_len.saturating_sub(option_size);
    }
    head.truncate(budget.max(8));
    let mut input_ids = vec![special.cls];
    input_ids.extend(head);
    input_ids.push(special.sep);
    let mut marker_positions = Vec::with_capacity(option_tokens.len());
    for option in option_tokens {
        marker_positions.push(input_ids.len());
        input_ids.extend(option);
    }
    input_ids.push(special.sep);
    let room = max_len.saturating_sub(input_ids.len().saturating_add(1));
    let state_tokens = tokenize(&state.replace(special.mask_text, " "))?;
    let state_slice = if truncate_left {
        // Upstream Python uses `state[-room:]`: when room is zero, `-0`
        // selects the whole state, then the complete sequence is truncated.
        let start = if room == 0 {
            0
        } else {
            state_tokens.len().saturating_sub(room)
        };
        &state_tokens[start..]
    } else {
        &state_tokens[..state_tokens.len().min(room)]
    };
    input_ids.extend_from_slice(state_slice);
    input_ids.push(special.sep);
    input_ids.truncate(max_len);
    if marker_positions
        .iter()
        .any(|&position| position >= input_ids.len())
    {
        return Err(DecisionError("option markers do not fit in max_len".into()));
    }
    Ok(PreparedSequence {
        input_ids,
        marker_positions,
        question_type,
    })
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
