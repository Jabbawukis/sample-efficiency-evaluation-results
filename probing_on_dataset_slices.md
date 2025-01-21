## For [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)

## Probing on training data slices

Train a model on the whole dataset and probe it after a slice of the training data has been processed.

Workflow: 

1. shuffle the dataset 
2. split it into slices
3. count fact occurrences on each slice
4. tokenize each slice to get the number of rows per slice
5. train the model on the shuffled and tokenized dataset and set a checkpoint approx. at the end of each slice (as save_steps)
6. probe each model checkpoint

Result: probing results for each model checkpoint, capturing the learning progress of the model.

- training script (for modified training arguments, see below): [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

#### Accuracy on checkpoints

For each checkpoint,
we count the number of facts with specific occurrences up until the slice seen by the model at said checkpoint.
The model checkpoint is probed and the accuracy depending on the number of occurrences up until the slice is calculated.

#### Cumulative distribution function (CDF) Analysis

For each checkpoint,
we optimize the CDF parameters ($\lambda$)
and derive the probability of the model
to answer a fact correctly given the number of occurrences of the fact in the training data up until the slice
seen by the model at said checkpoint.
The CDF functions are then plotted over each checkpoint.

$$f(x; \lambda) = 1 - e^{-\lambda x} , x\ge 0$$
$$\min_{\lambda} -\sum_{i=1}^{N} T_i*\log(f(occur(i); \lambda)) + (1 - T_i)*\log(f(occur(i);\lambda))$$

Where $occur(i)$ is the number of occurrences of the fact $i$ in the training data up until the slice 
seen by the model at the checkpoint.
## BEAR-big

### 1. gpt2_from_scratch
- Model: gpt2 (124M params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en_shuffeld)
- dataset shuffle seed: 42
- number of slices: 42
- per_device_train_batch_size: 32
- gradient_accumulation_steps: 8
- save_steps: 3650 (per slice num_rows_after_tokenized avg. ≈ 934,840 → 934,840 ÷ 8 ÷ 32 ≈ 3650)
- logging_steps: 3650


- link to slice info: [evaluation_on_slices](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to probing results: [probing results](probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices)


- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)
- link to CDF diagrams on checkpoints: [CDF on checkpoints](probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/cdf_log_likelihood_on_slices.png)

#### lm-evaluation-harness scores (final model)
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.5193|±  | 0.014|
|wsc273|       1 |none  |     0|acc   |↑  |0.5165|±  |0.0303|
|lambada_standard|       1 |none  |     0|acc       |↑  |  0.1558|±  | 0.0051|
|lambada_standard|       1 |none  |     0|perplexity|↓  |822.1627|±  |42.0769|
|pile_10k|       1 |none  |     0|bits_per_byte  |↓  |    2.0200|±  |   N/A|
|pile_10k|       1 |none  |     0|byte_perplexity|↓  |    4.0560|±  |   N/A|
|pile_10k|       1 |none  |     0|word_perplexity|↓  |11840.3982|±  |   N/A|

### 2. xlstm_from_scratch

- Model: xLSTM (163.8M params)
- repo (model checkpoints as branches): [J4bb4wukis/xlstm_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/xlstm_wikipedia_en_shuffeld)
- dataset shuffle seed: 42
- number of slices: 42
- per_device_train_batch_size: 32
- gradient_accumulation_steps: 8
- save_steps: 3650 (per slice num_rows_after_tokenized avg. ≈ 934,840 → 934,840 ÷ 8 ÷ 32 ≈ 3650)
- logging_steps: 3650


- link to slice info: [evaluation_on_slices](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to probing results: [probing results](probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices)


- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)
- link to CDF diagrams on checkpoints: [CDF on checkpoints](probing_results/BEAR-big/xlstm_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/cdf_log_likelihood_on_slices.png)

#### lm-evaluation-harness scores (final model)
|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.5043|±  |0.0141|
|wsc273|      1|none  |     0|acc   |↑  |0.5495|±  |0.0302|
|lambada_standard|      1|none  |     0|acc       |↑  |   0.0935|±  | 0.0041|
|lambada_standard|1|none  |     0|perplexity|↓  |1536.1172|±  |74.8833|