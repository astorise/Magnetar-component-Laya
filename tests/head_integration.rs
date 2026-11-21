use magnetar_component_laya::{
    Calibrator, QuestionType, act_probability,
    head::{DecisionHead, LayerNorm, Linear, TransformerLayer},
};
use serde::Deserialize;

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
        let decision = calibrator
            .decide(
                qtype,
                &result
                    .option_logits
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let probabilities = decision
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
}
