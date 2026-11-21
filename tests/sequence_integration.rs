use magnetar_component_laya::{ChoiceOption, Question, SpecialTokens, prepare_sequence};

const TOKENS: SpecialTokens<'static> = SpecialTokens {
    cls: 101,
    sep: 102,
    mask: 103,
    mask_text: "[MASK]",
};

fn encode(text: &str) -> Result<Vec<u32>, &'static str> {
    Ok(text
        .chars()
        .map(|character| character as u32 + 1000)
        .collect())
}

fn verify(case: &serde_json::Value, question: &Question<'_>) {
    let prepared = prepare_sequence(
        question,
        case["state"].as_str().unwrap(),
        TOKENS,
        case["max_len"].as_u64().unwrap() as usize,
        case["head_max_len"].as_u64().unwrap() as usize,
        case["truncate_left"].as_bool().unwrap(),
        encode,
    )
    .unwrap();
    let ids: Vec<u32> = serde_json::from_value(case["input_ids"].clone()).unwrap();
    let positions: Vec<usize> = serde_json::from_value(case["marker_positions"].clone()).unwrap();
    assert_eq!(prepared.input_ids, ids);
    assert_eq!(prepared.marker_positions, positions);
}

#[test]
fn sequence_layout_matches_pinned_upstream_with_a_deterministic_tokenizer() {
    let cases: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/sequence_reference.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let instructions = case["instructions"].as_str().unwrap();
        match case["type"].as_str().unwrap() {
            "choice" => {
                let options: Vec<_> = case["choices_ordered"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|entry| ChoiceOption {
                        label: entry[0].as_str().unwrap(),
                        description: entry[1].as_str(),
                    })
                    .collect();
                verify(
                    case,
                    &Question::Choice {
                        instructions,
                        options: &options,
                    },
                );
            }
            "score" => {
                let levels: Vec<_> = case["criteria"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| item.as_str().unwrap())
                    .collect();
                verify(
                    case,
                    &Question::Score {
                        instructions,
                        levels: &levels,
                    },
                );
            }
            "noul" => verify(
                case,
                &Question::Noul {
                    instructions,
                    when_false: None,
                    when_true: None,
                },
            ),
            kind => panic!("unexpected question type {kind}"),
        }
    }
}

#[test]
fn refuses_missing_options_truncated_markers_and_tokenizer_errors() {
    let empty = Question::Choice {
        instructions: "x",
        options: &[],
    };
    assert!(prepare_sequence(&empty, "", TOKENS, 100, 80, false, encode).is_err());
    let options = [ChoiceOption {
        label: "a",
        description: None,
    }];
    let valid = Question::Choice {
        instructions: "x",
        options: &options,
    };
    assert!(prepare_sequence(&valid, "", TOKENS, 1, 80, false, encode).is_err());
    assert!(
        prepare_sequence(&valid, "", TOKENS, 100, 80, false, |_| Err::<Vec<u32>, _>(
            "bad tokenizer"
        ))
        .is_err()
    );
    let invalid_tokens = SpecialTokens {
        mask_text: "",
        ..TOKENS
    };
    assert!(prepare_sequence(&valid, "", invalid_tokens, 100, 80, false, encode).is_err());
}
