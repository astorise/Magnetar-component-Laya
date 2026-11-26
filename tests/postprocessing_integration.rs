use magnetar_component_laya::{Answer, Calibrator, QuestionType, act_probability};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual {actual:.16}, expected {expected:.16}"
    );
}

#[test]
fn published_checkpoint_temperatures_match_upstream_decision_reference() {
    let calibrator =
        Calibrator::from_config(include_bytes!("fixtures/rl_agent_config.json")).unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/postprocessing_reference.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let kind = match case["kind"].as_str().unwrap() {
            "choice" => QuestionType::Choice,
            "score" => QuestionType::Score,
            "noul" => QuestionType::Noul,
            other => panic!("unexpected test kind {other}"),
        };
        let logits: Vec<f64> = case["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        close(
            calibrator.temperature(kind, logits.len()),
            case["temperature"].as_f64().unwrap(),
        );
        let decision = calibrator.decide(kind, &logits).unwrap();
        for (actual, expected) in decision
            .probabilities
            .iter()
            .zip(case["probabilities"].as_array().unwrap())
        {
            close(*actual, expected.as_f64().unwrap());
        }
        match decision.answer {
            Answer::Choice(index) => assert_eq!(index as u64, case["answer"].as_u64().unwrap()),
            Answer::Score(score) | Answer::Noul(score) => {
                close(score, case["answer"].as_f64().unwrap())
            }
        }
        if kind == QuestionType::Noul {
            assert_eq!(decision.confidence, None);
        } else {
            close(
                decision.confidence.unwrap(),
                case["confidence"].as_f64().unwrap(),
            );
        }
    }
}

#[test]
fn absent_temperatures_default_to_one_and_ties_choose_first() {
    let calibrator = Calibrator::from_config(b"{}").unwrap();
    let decision = calibrator
        .decide(QuestionType::Choice, &[1000.0, 1000.0])
        .unwrap();
    assert_eq!(decision.answer, Answer::Choice(0));
    assert_eq!(decision.probabilities, vec![0.5, 0.5]);
    close(decision.confidence.unwrap(), 0.0);
    close(act_probability([1000.0, 1000.0]).unwrap(), 0.5);
}

#[test]
fn rejects_invalid_temperatures_logits_and_option_counts() {
    assert!(Calibrator::from_config(br#"{"temperature":[1,0,1]}"#).is_err());
    assert!(Calibrator::from_config(br#"{"temperature":[1,1]}"#).is_err());
    assert!(Calibrator::from_config(br#"{"temperature_by_options":{"choice:2":-1}}"#).is_err());
    let calibrator = Calibrator::from_config(b"{}").unwrap();
    assert!(calibrator.decide(QuestionType::Choice, &[]).is_err());
    assert!(calibrator.decide(QuestionType::Noul, &[0.0]).is_err());
    assert!(
        calibrator
            .decide(QuestionType::Noul, &[0.0, 1.0, 2.0])
            .is_err()
    );
    assert!(calibrator.decide(QuestionType::Score, &[f64::NAN]).is_err());
    assert!(act_probability([f64::INFINITY, 0.0]).is_err());
}
