use magnetar_component_laya::{
    Answer, Calibrator, QuestionType, act_probability,
    head::{
        CheckpointTensor, DecisionHead, LayerNorm, Linear, TransformerLayer,
        checkpoint_tensor_shapes,
    },
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Case {
    hidden: usize,
    tokens: usize,
    mask: Vec<u8>,
    markers: Vec<usize>,
    qtype: usize,
    logits: Vec<f32>,
    action_logits: Vec<f32>,
    probabilities: Vec<f32>,
    action_probability: f32,
}

fn pattern(name: &str, count: usize) -> Vec<f32> {
    let seed: usize = name
        .bytes()
        .enumerate()
        .map(|(index, byte)| (index + 1) * byte as usize)
        .sum::<usize>()
        % 97;
    (0..count)
        .map(|index| {
            let mut value = ((index * 17 + seed) % 101) as f32 - 50.0;
            value /= 200.0;
            if name.contains("norm") && name.ends_with(".weight") {
                value += 1.0;
            }
            value
        })
        .collect()
}

fn linear(name: &str, input: usize, output: usize) -> Linear {
    Linear {
        weight: pattern(&format!("{name}.weight"), input * output),
        bias: pattern(&format!("{name}.bias"), output),
    }
}

fn norm(name: &str, hidden: usize) -> LayerNorm {
    LayerNorm {
        weight: pattern(&format!("{name}.weight"), hidden),
        bias: pattern(&format!("{name}.bias"), hidden),
    }
}

fn head(d: usize) -> DecisionHead {
    let layers = (0..2)
        .map(|index| {
            let prefix = format!("head.layers.{index}");
            TransformerLayer {
                norm1: norm(&format!("{prefix}.norm1"), d),
                norm2: norm(&format!("{prefix}.norm2"), d),
                qkv: Linear {
                    weight: pattern(&format!("{prefix}.self_attn.in_proj_weight"), 3 * d * d),
                    bias: pattern(&format!("{prefix}.self_attn.in_proj_bias"), 3 * d),
                },
                attention_output: linear(&format!("{prefix}.self_attn.out_proj"), d, d),
                feed_forward_in: linear(&format!("{prefix}.linear1"), d, 4 * d),
                feed_forward_out: linear(&format!("{prefix}.linear2"), 4 * d, d),
            }
        })
        .collect();
    DecisionHead {
        hidden: d,
        type_embedding: pattern("type_emb.weight", 3 * d),
        layers,
        scorer_norm: norm("scorer.0", d),
        scorer_in: linear("scorer.1", d, d),
        scorer_out: linear("scorer.3", d, 1),
        action_in: linear("act_head.0", d + 4, 256),
        action_out: linear("act_head.2", 256, 2),
    }
}

fn checkpoint_tensors(hidden: usize) -> BTreeMap<String, CheckpointTensor> {
    checkpoint_tensor_shapes(hidden, 2, 2)
        .unwrap()
        .into_iter()
        .map(|(name, shape)| {
            let values = pattern(&name, shape.iter().product());
            (name, CheckpointTensor { shape, values })
        })
        .collect()
}

#[test]
fn checkpoint_head_layout_matches_the_pinned_f16_header() {
    #[derive(Deserialize)]
    struct Header {
        header_sha256: String,
        tensors: BTreeMap<String, Descriptor>,
    }
    #[derive(Deserialize)]
    struct Descriptor {
        dtype: String,
        shape: Vec<usize>,
    }
    let header: Header =
        serde_json::from_str(include_str!("fixtures/checkpoint-head-header.json")).unwrap();
    assert_eq!(
        header.header_sha256,
        "3a42d8e7a96d5aa22c5bc9475d7ff092e32a832ae7c6973688f66f7a84048c04"
    );
    assert!(header.tensors.values().all(|tensor| tensor.dtype == "F16"));
    let actual = header
        .tensors
        .into_iter()
        .map(|(name, tensor)| (name, tensor.shape))
        .collect();
    assert_eq!(checkpoint_tensor_shapes(1024, 2, 2).unwrap(), actual);
}

#[test]
fn checkpoint_head_mapping_matches_pytorch_and_rejects_bad_tensors() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/head_reference.json")).unwrap();
    let case = &cases[0];
    let tensors = checkpoint_tensors(case.hidden);
    let loaded = DecisionHead::from_checkpoint_tensors(case.hidden, 2, 2, tensors.clone()).unwrap();
    let result = loaded
        .forward(
            &pattern("encoder_output", case.tokens * case.hidden),
            case.tokens,
            &case
                .mask
                .iter()
                .map(|&value| value != 0)
                .collect::<Vec<_>>(),
            &case.markers,
            QuestionType::Choice,
        )
        .unwrap();
    close(&result.option_logits, &case.logits, "loaded option logits");
    close(
        &result.action_logits,
        &case.action_logits,
        "loaded action logits",
    );

    let mut missing = tensors.clone();
    missing.remove("head.layers.0.self_attn.in_proj_weight");
    assert!(DecisionHead::from_checkpoint_tensors(case.hidden, 2, 2, missing).is_err());
    let mut malformed = tensors.clone();
    malformed.get_mut("scorer.1.weight").unwrap().shape = vec![1, case.hidden * case.hidden];
    assert!(DecisionHead::from_checkpoint_tensors(case.hidden, 2, 2, malformed).is_err());
    let mut nonfinite = tensors;
    nonfinite.get_mut("act_head.2.bias").unwrap().values[0] = f32::NAN;
    assert!(DecisionHead::from_checkpoint_tensors(case.hidden, 2, 2, nonfinite).is_err());
    assert!(checkpoint_tensor_shapes(usize::MAX, 2, 2).is_err());
}

#[test]
#[ignore = "requires decoded weights from the pinned Laya checkpoint"]
fn decoded_checkpoint_head_matches_upstream_pytorch() {
    #[derive(Deserialize)]
    struct Reference {
        cases: Vec<RealCase>,
    }
    #[derive(Deserialize)]
    struct RealCase {
        mask: Vec<u8>,
        markers: Vec<usize>,
        qtype: usize,
        logits: Vec<f32>,
        action_logits: Vec<f32>,
        probabilities: Vec<f32>,
        action_probabilities: Vec<f32>,
        answer: f64,
    }
    let path = std::env::var_os("MAGNETAR_LAYA_HEAD_F32_DIR")
        .expect("set MAGNETAR_LAYA_HEAD_F32_DIR to the generated decoded weights");
    let mut tensors = BTreeMap::new();
    for (name, shape) in checkpoint_tensor_shapes(1024, 2, 2).unwrap() {
        let bytes = std::fs::read(std::path::Path::new(&path).join(format!("{name}.f32"))).unwrap();
        assert_eq!(bytes.len(), shape.iter().product::<usize>() * 4, "{name}");
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        tensors.insert(name, CheckpointTensor { shape, values });
    }
    let model = DecisionHead::from_checkpoint_tensors(1024, 2, 2, tensors).unwrap();
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/real-head-reference.json")).unwrap();
    let calibrator =
        Calibrator::from_config(include_bytes!("fixtures/rl_agent_config.json")).unwrap();
    for case in reference.cases {
        let kind = match case.qtype {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            _ => panic!("unexpected question type"),
        };
        let result = model
            .forward(
                &pattern("encoder_output", 4 * 1024),
                4,
                &case
                    .mask
                    .iter()
                    .map(|&value| value != 0)
                    .collect::<Vec<_>>(),
                &case.markers,
                kind,
            )
            .unwrap();
        let decision = result.decide(&calibrator, kind).unwrap();
        match decision.decision.answer {
            Answer::Choice(index) => assert_eq!(index as f64, case.answer),
            Answer::Score(value) => assert!((value - case.answer).abs() < 1e-5),
            Answer::Noul(_) => panic!("unexpected answer type"),
        }
        let probabilities = decision
            .decision
            .probabilities
            .iter()
            .map(|&v| v as f32)
            .collect();
        let action_probabilities = decision
            .action_probabilities
            .iter()
            .map(|&v| v as f32)
            .collect();
        for (label, actual, expected) in [
            ("option", result.option_logits, case.logits),
            ("action", result.action_logits, case.action_logits),
            ("option probability", probabilities, case.probabilities),
            (
                "action probability",
                action_probabilities,
                case.action_probabilities,
            ),
        ] {
            assert_eq!(actual.len(), expected.len());
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("{label} maximum absolute error: {max_error}");
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 3e-4 + 1e-5 * expected.abs(),
                    "{label}[{index}]: {actual} != {expected}"
                );
            }
        }
    }
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 2e-6 + 1e-6 * b.abs(),
            "{label}[{index}]: {a} != {b}"
        );
    }
}

#[test]
fn learned_head_matches_pinned_pytorch_with_padding_and_multiple_heads() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/head_reference.json")).unwrap();
    let calibrator =
        Calibrator::from_config(include_bytes!("fixtures/rl_agent_config.json")).unwrap();
    for case in cases {
        let qtype = match case.qtype {
            0 => QuestionType::Choice,
            1 => QuestionType::Score,
            2 => QuestionType::Noul,
            _ => panic!("unknown question type"),
        };
        let result = head(case.hidden)
            .forward(
                &pattern("encoder_output", case.tokens * case.hidden),
                case.tokens,
                &case
                    .mask
                    .iter()
                    .map(|&value| value != 0)
                    .collect::<Vec<_>>(),
                &case.markers,
                qtype,
            )
            .unwrap();
        close(&result.option_logits, &case.logits, "option logits");
        close(&result.action_logits, &case.action_logits, "action logits");
        let decision = result.decide(&calibrator, qtype).unwrap();
        assert_eq!(decision.option_scores, result.option_logits);
        assert_eq!(decision.action_scores, result.action_logits);
        let probabilities = decision
            .decision
            .probabilities
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        close(&probabilities, &case.probabilities, "option probabilities");
        let action = act_probability([
            result.action_logits[0] as f64,
            result.action_logits[1] as f64,
        ])
        .unwrap();
        close(
            &[action as f32],
            &[case.action_probability],
            "action probability",
        );
        close(
            &[decision.action_probabilities[0] as f32],
            &[case.action_probability],
            "typed action probability",
        );
    }
}

#[test]
fn learned_head_rejects_bad_shapes_masks_markers_and_weights() {
    let mut model = head(8);
    let input = pattern("encoder_output", 24);
    assert!(
        model
            .forward(
                &input,
                3,
                &[true, true, false],
                &[1, 2],
                QuestionType::Choice
            )
            .is_err()
    );
    assert!(
        model
            .forward(&input, 3, &[false; 3], &[0, 1], QuestionType::Choice)
            .is_err()
    );
    assert!(
        model
            .forward(&input, 3, &[true; 3], &[1], QuestionType::Choice)
            .is_err()
    );
    assert!(
        model
            .forward(&input[..16], 3, &[true; 3], &[0, 1], QuestionType::Choice)
            .is_err()
    );
    model.layers[0].qkv.weight[0] = f32::NAN;
    assert!(
        model
            .forward(&input, 3, &[true; 3], &[0, 1], QuestionType::Choice)
            .is_err()
    );
    let mut overflow = head(8);
    overflow.layers[0].feed_forward_in.weight.fill(f32::MAX);
    assert!(
        overflow
            .forward(&input, 3, &[true; 3], &[0, 1], QuestionType::Choice)
            .is_err()
    );
    let calibrator = Calibrator::from_config(b"{}").unwrap();
    let missing_actions = magnetar_component_laya::head::HeadOutput {
        option_logits: vec![0.0, 1.0],
        action_logits: vec![0.0],
    };
    assert!(
        missing_actions
            .decide(&calibrator, QuestionType::Choice)
            .is_err()
    );
    let non_finite_actions = magnetar_component_laya::head::HeadOutput {
        option_logits: vec![0.0, 1.0],
        action_logits: vec![0.0, f32::NAN],
    };
    assert!(
        non_finite_actions
            .decide(&calibrator, QuestionType::Choice)
            .is_err()
    );
}
