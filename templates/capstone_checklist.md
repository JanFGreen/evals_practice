# Capstone project: checklist

## Part A: a full suite for Nordwind Assist
- [ ] The golden set has at least 60 cases, including 20 new ones from error review or from `synth` after manual verification
- [ ] `audit` shows no duplicates and no cases without checks
- [ ] Oracle about 100%, null about 0% (`sanity`)
- [ ] Correctness judge calibrated: TPR and TNR on the test split ≥ 0.85, disagreements described
- [ ] Faithfulness and retriever metrics added to the run; every failure attributed to the retriever or to generation
- [ ] Red teaming: 0 canary leaks, 0 followed injections, 0 isolation violations
- [ ] Agent: pass^3 computed for every task, low pass^3 tasks described
- [ ] Baseline saved in `evals/baseline`, the CI gate catches a deliberately introduced regression
- [ ] Version v3 (or your own) compared with v2 on train/test, with a confidence interval for the difference
- [ ] Eval card (`templates/eval_card.md`) filled in

## Part B (optional): transfer to your own system
- [ ] A chosen system (a work application or your own project) and one sentence about the decision the eval should support
- [ ] 30 cases from real traffic or from domain experts, with references written by humans
- [ ] Deterministic checks for this system's requirements (format, citations, forbidden content)
- [ ] A retriever eval on the real index, without LLM calls
- [ ] A data isolation test between customers (if the system is multi-tenant)
- [ ] A judge calibrated on at least 40 human labels
- [ ] One CI run with a gate
