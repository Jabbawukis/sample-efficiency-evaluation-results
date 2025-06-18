# Sample Efficiency Evaluation Results

## Overview

For every model:
- extract probing results in the `probing_results_on_checkpoints` directory into a `checkpoint_extracted` directory within the same 
parent directory (only on BEAR-big)
- extract the `increasing_occurrences_in_slices.tar.xz` file in the same parent directory

Example:
```
gpt2_209m
└── wikimedia_wikipedia_20231101_en
    ├── accuracy_statistics_final_gpt2_209m_bear_big.png
    ├── accuracy_statistics_final_gpt2_209m_bear_big.pdf
    ├── evaluation_on_slices
    │   ├── combined_accuracy_plots_grid.png
    │   ├── correct_answer_probability_optimized_params
    │   │   ├── nll_on_slices_bear_big.png
    │   │   ├── nll_on_slices_bear_big.pdf
    │   │   ├── nll_on_slices.json
    │   │   └── optimized_params
    │   │       ├── cdf_optimized_lambdas.json
    │   │       └── psf_optimized_alphas.json
    │   ├── increasing_occurrences_in_slices.json  <--- extracted from increasing_occurrences_in_slices.tar.xz
    │   ├── increasing_occurrences_in_slices.tar.xz
    │   └── probing_results_on_checkpoints
    │       ├── checkpoint-102200.tar.gz
    │       ├── checkpoint-105850.tar.gz
    ...           ...
    │       └── checkpoint_extracted <--- extracted probing results
    │           ├── checkpoint-102200
    │           ├── checkpoint-105850
    ...           ...
    └── probing_scores_final_model
        ├── metadata_results.json
        ├── P101_results.jsonl
        ├── P103_results.jsonl
        ...

## Citation

```
    @inproceedings{
    anonymous2025from,
    title={From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts},
    author={Anonymous},
    booktitle={First Workshop on Large Language Model Memorization},
    year={2025},
    url={https://openreview.net/forum?id=iXHpdSGd8o}
    }
