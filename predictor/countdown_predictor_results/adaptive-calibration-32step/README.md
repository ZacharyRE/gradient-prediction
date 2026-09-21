# Adaptive Calibration: Configuration and Evidence

**[Read the main Countdown report →](../README.md#dynamic-calibration)**

This existing bundle contains the completed single-layer 32-step study. The parent topic README maintains the method, comparison table, and conclusions; this page identifies the supporting files.

| File | Contents |
|---|---|
| [protocol.json](protocol.json) | Model, predictor, 4-of-32 calibration, fixed 16-example validation, epoch selection, optimizers, and evaluation |
| [results/test_accuracy.csv](results/test_accuracy.csv) | All 37 complete 2,048-question test evaluations, including baseline and individual seeds |
| [results/dev_accuracy.csv](results/dev_accuracy.csv) | Accuracy-development curves, separate from the gradient-validation set |
| [results/calibration_choices.csv](results/calibration_choices.csv) | All 96 selected epochs, validation scores, batch indices, and four-example calibration subsets |
| [results/gradient_metrics.csv](results/gradient_metrics.csv) | All-step activation and LoRA reconstruction metrics; current-batch and held-out diagnostics |
| [figures/accuracy.png](figures/accuracy.png) / [PDF](figures/accuracy.pdf) | Test comparison plot |
| [results/test_audit.json](results/test_audit.json) | Full question counts, preserved inputs, and summary checks |
| [source/](source/README.md) | Source snapshots and their reproduction boundary |
| [provenance.json](provenance.json) | Original artifact and published-source SHA256 identities |

All-layer downstream experiments are outside this completed bundle. Full inputs are retained; 2,048 new tokens is the output-generation cap.
