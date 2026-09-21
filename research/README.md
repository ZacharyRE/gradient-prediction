# SFT Research Archive

[Project overview](../README.md) · [Results index](../docs/RESULTS.md) · [Current predictor research](../predictor/README.md)

These published studies examine LoRA SFT degradation, supervision targets, and generalization. They provide background for interpreting the later predictor-driven update experiments. Their models, training budgets, and evaluation settings are study-specific.

| Study | Main question | Reading |
|---|---|---|
| September 8: training diagnosis | Which training and evaluation choices explain the observed SFT degradation? | [Study entry](sft_diagnosis_20260908/README.md) · [Report](sft_diagnosis_20260908/REPORT.md) |
| September 9: generalization and supervision | Can revised targets recover accuracy and produce stable gains across seeds and tasks? | [Study entry](sft_generalization_20260909/README.md) · [Report](sft_generalization_20260909/REPORT.md) |

Start with each study's report, then follow its methods, result, figure, and audit references as needed. The September 9 study extends the earlier diagnosis; it does not establish a general failure of LoRA SFT.

The dated directories are evidence archives. Their report files, statistical records, and source snapshots retain their recorded provenance. Reproduction requires the assets and dependencies described in each study; historical queue/watch scripts are not general launch instructions. Current Countdown results live in the [Countdown topic](../predictor/countdown_predictor_results/README.md).
