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

#### Accuracy on checkpoints

For each checkpoint,
we count the number of facts with specific occurrences up until the slice seen by the model at said checkpoint.
The model checkpoint is probed and the accuracy depending on the number of occurrences up until the slice is calculated.

#### Correct Answer Probability Analysis

For each checkpoint, derive the probability of the model
to answer a fact correctly given the number of occurrences of the fact in the training data up until the slice
seen by the model at said checkpoint. 

We optimize ...

$$f(x; \lambda) = 1 - e^{-\lambda x} , x\ge 0$$
$$mapped(i;x;\lambda) = min\_prob(i) + (max\_prob(i) - min\_prob(i)) * f(x; \lambda) $$

The parameters are optimized by minimizing the negative log-likelihood of the model's predictions up until the slice seen by the model:

$$\min_{\lambda; max\_prob}LL(\lambda; max\_prob) -\sum_{i=1}^{N} T_i*\log(mapped(i;occur(i); \lambda)) + (1 - T_i)*\log(mapped(i;occur(i);\lambda))$$

Where:
- $occur(i)$ is the number of occurrences of the fact $i$ in the training data up until the slice 
seen by the model at the checkpoint. $T_i$ is the target value for the fact $i$ (1 if the model answered correctly, 0 otherwise).
- $mapped(i;occur(i);\lambda)$ is the probability of the model to answer a fact $i$ correctly given the number of occurrences $occur(i)$ of the fact,
mapped to the interval $[min\_prob(i), max\_prob(i)]$.
- $min\_prob(i)$ is the minimum possible probability a model can achieve on a fact $i$.

After the optimization, models with a higher ..., and thus a steeper curve for the plotted probability function
are considered to have a better sample efficiency. The probability of the model to answer a fact correctly for
low occurrences of a fact in the training data is higher.

## BEAR-big

- fact matching results on slices: [fact_matching_results](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices/)
- link to slice info: [evaluation_on_slices](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- dataset shuffle seed: 42
- number of slices: 42
- per_device_train_batch_size: 32
- gradient_accumulation_steps: 8
- save_steps: 3650 (per slice num_rows_after_tokenized avg. ≈ 934,840 → 934,840 ÷ 8 ÷ 32 ≈ 3650)
- logging_steps: 3650

### 1. gpt2_124m
- Model: GPT2 (124M params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_124m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_124m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)


- link to probing results: [probing results](probing_results/BEAR-big/gpt2_124m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_124m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

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

### 2. xlstm_247m

- Model: xLSTM (247M params with GPT2 tokenizer vocab size, else, 163.8M params if using the author config)
- repo (model checkpoints as branches): [J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/xLSTM/wikimedia_wikipedia_20231101_en/train.py)

- link to probing results: [probing results](probing_results/BEAR-big/xlstm_247m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/xlstm_247m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### lm-evaluation-harness scores (final model)
|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.5043|±  |0.0141|
|wsc273|      1|none  |     0|acc   |↑  |0.5495|±  |0.0302|
|lambada_standard|      1|none  |     0|acc       |↑  |   0.0935|±  | 0.0041|
|lambada_standard|1|none  |     0|perplexity|↓  |1536.1172|±  |74.8833|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |  1.4805|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |  2.7904|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |966.7574|±  |   N/A|


### 3. mamba2_172m
- Model: Mamba2 (172M params with GPT2 tokenizer vocab size, else, 130M params if using the author config)
- repo (model checkpoints as branches): [J4bb4wukis/mamba2_127m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/mamba2_127m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/Mamba2/wikimedia_wikipedia_20231101_en/train.py)


- link to probing results: [probing results](probing_results/BEAR-big/mamba2_172m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/mamba2_172m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### lm-evaluation-harness scores (final model)
|  Tasks   |Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|     1 |none  |     0|acc   |↑  |0.502|±  |0.0141|
|wsc273|     1 |none  |     0|acc   |↑  |0.5092|±  |0.0303|
|lambada_standard|     1 |none  |     0|acc       |↑  |   0.0768|±  |  0.0037|
|lambada_standard|     1 |none  |     0|perplexity|↓  |2183.7652|±  |109.3855|
|pile_10k|     1 |none  |     0|bits_per_byte  |↓  |   1.5435|±  |   N/A|
|pile_10k|     1 |none  |     0|byte_perplexity|↓  |   2.9149|±  |   N/A|
|pile_10k|     1 |none  |     0|word_perplexity|↓  |1295.2241|±  |   N/A|


### 4. gpt2_209m
- Model: GPT2 (209M params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_209m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_209m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

###### Adjusted training parameters
- n_head=16
- n_layer=24

- link to probing results: [probing results](probing_results/BEAR-big/gpt2_209m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_209m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### lm-evaluation-harness scores (final model)
|  Tasks   |  Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|---------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|        1 |none  |     0|acc   |↑  |0.5036|±  |0.0141|
|wsc273|        1 |none  |     0|acc   |↑  |0.5311|±  |0.0303|
|lambada_standard|        1 |none  |     0|acc       |↑  |  0.1663|±  | 0.0052|
|lambada_standard|        1 |none  |     0|perplexity|↓  |652.0058|±  |33.1575|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |    2.0620|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |    4.1758|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |14389.4299|±  |   N/A|