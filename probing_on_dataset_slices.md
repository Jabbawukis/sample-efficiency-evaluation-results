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

## Accuracy on checkpoints

For each checkpoint,
we count the number of facts with specific occurrences up until the slice seen by the model at said checkpoint.
The model checkpoint is probed and the accuracy depending on the number of occurrences up until the slice is calculated.

### BEAR-big

- fact matching results on slices: [fact_matching_results](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices/)
- dataset shuffle seed: 42
- number of slices: 42
- per_device_train_batch_size: 32
- gradient_accumulation_steps: 8
- save_steps: 3650 (per slice num_rows_after_tokenized avg. ≈ 934,840 → 934,840 ÷ 8 ÷ 32 ≈ 3650)
- logging_steps: 3650

Training is 3650 steps per slice, 42 slices,
153,300 steps in total.
We approximate the fact occurrences within each slice by slicing the raw data into 42 slices
and then running the fact matching process over the slices to get the approximate number of facts per slice.
Training the model on the 
entire dataset and saving each checkpoint at 3650 steps,
we get the state of the model after seeing approximately the number of facts determined
by the fact matching on the raw data.
We also evaluate the final model as the last checkpoint ends at 153,300 steps
and the entire training has 153,372 in total with 72 steps deviations.

#### 1. gpt2_124m
- Model: GPT2 (124M params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_124m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_124m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)


- link to probing results: [probing results](probing_results/BEAR-big/gpt2_124m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_124m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.5193|±  | 0.014|
|wsc273|       1 |none  |     0|acc   |↑  |0.5165|±  |0.0303|
|lambada_standard|       1 |none  |     0|acc       |↑  |  0.1558|±  | 0.0051|
|lambada_standard|       1 |none  |     0|perplexity|↓  |822.1627|±  |42.0769|
|pile_10k|       1 |none  |     0|bits_per_byte  |↓  |    2.0200|±  |   N/A|
|pile_10k|       1 |none  |     0|byte_perplexity|↓  |    4.0560|±  |   N/A|
|pile_10k|       1 |none  |     0|word_perplexity|↓  |11840.3982|±  |   N/A|

#### 2. xlstm_247m

- Model: xLSTM (247M params with GPT2 tokenizer vocab size, else, 163.8M params if using the author config)
- repo (model checkpoints as branches): [J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/xlstm_247m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/xLSTM/wikimedia_wikipedia_20231101_en/train.py)

- link to probing results: [probing results](probing_results/BEAR-big/xlstm_247m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/xlstm_247m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.5043|±  |0.0141|
|wsc273|      1|none  |     0|acc   |↑  |0.5495|±  |0.0302|
|lambada_standard|      1|none  |     0|acc       |↑  |   0.0935|±  | 0.0041|
|lambada_standard|1|none  |     0|perplexity|↓  |1536.1172|±  |74.8833|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |  1.4805|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |  2.7904|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |966.7574|±  |   N/A|


##### 3. mamba2_172m
- Model: Mamba2 (172M params with GPT2 tokenizer vocab size, else, 130M params if using the author config)
- repo (model checkpoints as branches): [J4bb4wukis/mamba2_172m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/mamba2_172m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/Mamba2/wikimedia_wikipedia_20231101_en/train.py)


- link to probing results: [probing results](probing_results/BEAR-big/mamba2_172m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/mamba2_172m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   |Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|     1 |none  |     0|acc   |↑  |0.502|±  |0.0141|
|wsc273|     1 |none  |     0|acc   |↑  |0.5092|±  |0.0303|
|lambada_standard|     1 |none  |     0|acc       |↑  |   0.0768|±  |  0.0037|
|lambada_standard|     1 |none  |     0|perplexity|↓  |2183.7652|±  |109.3855|
|pile_10k|     1 |none  |     0|bits_per_byte  |↓  |   1.5435|±  |   N/A|
|pile_10k|     1 |none  |     0|byte_perplexity|↓  |   2.9149|±  |   N/A|
|pile_10k|     1 |none  |     0|word_perplexity|↓  |1295.2241|±  |   N/A|


#### 4. gpt2_209m
- Model: GPT2 (209M params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_209m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_209m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

###### Adjusted training parameters
- n_head=16
- n_layer=24


- link to probing results: [probing results](probing_results/BEAR-big/gpt2_209m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_209m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   |  Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|---------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|        1 |none  |     0|acc   |↑  |0.5036|±  |0.0141|
|wsc273|        1 |none  |     0|acc   |↑  |0.5311|±  |0.0303|
|lambada_standard|        1 |none  |     0|acc       |↑  |  0.1663|±  | 0.0052|
|lambada_standard|        1 |none  |     0|perplexity|↓  |652.0058|±  |33.1575|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |    2.0620|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |    4.1758|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |14389.4299|±  |   N/A|

#### 5. gpt2_355m
- Model: GPT2 (350m params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_355m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_355m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

###### Adjusted training parameters
- n_embd=1024
- n_head=16
- n_layer=24


- link to probing results: [probing results](probing_results/BEAR-big/gpt2_355m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_355m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.5162|±  | 0.014|
|wsc273|       1 |none  |     0|acc   |↑  |0.5458|±  |0.0302|
|lambada_standard|       1 |none  |     0|acc       |↑  |  0.1644|±  | 0.0052|
|lambada_standard|       1 |none  |     0|perplexity|↓  |592.8151|±  |29.6474|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |    2.1101|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |    4.3171|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |17984.4641|±  |   N/A|


#### 6. mamba2_432m
- Model: Mamba2 (432m params with GPT2 tokenizer vocab size)
- repo (model checkpoints as branches): [J4bb4wukis/mamba2_432m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/mamba2_432m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/Mamba2/wikimedia_wikipedia_20231101_en/train.py)

###### Adjusted training parameters
- hidden_size=1024
- num_heads=32
- num_hidden_layers=48
- state_size=32
- head_dim=64


- link to probing results: [probing results](probing_results/BEAR-big/mamba2_432m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/mamba2_432m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.5067|±  |0.0141|
|wsc273|       1 |none  |     0|acc   |↑  |0.5458|±  |0.0302|
|lambada_standard|       1 |none  |     0|acc       |↑  |   0.0788|±  | 0.0038|
|lambada_standard|       1 |none  |     0|perplexity|↓  |1594.1999|±  |77.5151|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |   1.5115|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |   2.8511|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |1116.7870|±  |   N/A|


#### 7. xlstm_406m

- Model: xLSTM (406M params with GPT2 tokenizer vocab size)
- repo (model checkpoints as branches): [J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/xlstm_406m_wikipedia_en_shuffeld)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/xLSTM/wikimedia_wikipedia_20231101_en/train.py)

- link to probing results: [probing results](probing_results/BEAR-big/xlstm_406m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/xlstm_406m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

##### lm-evaluation-harness scores (final model)
|  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.5146|±  | 0.014|
|wsc273|      1|none  |     0|acc   |↑  |0.5055|±  |0.0303|
|lambada_standard|      1|none  |     0|acc       |↑  |  0.1197|±  | 0.0045|
|lambada_standard|      1|none  |     0|perplexity|↓  |739.1623|±  |34.8244|
|pile_10k|      1|none  |     0|bits_per_byte  |↓  |  1.4628|±  |   N/A|
|pile_10k|      1|none  |     0|byte_perplexity|↓  |  2.7564|±  |   N/A|
|pile_10k|      1|none  |     0|word_perplexity|↓  |890.4901|±  |   N/A|

### BEAR-small

- fact matching results on slices: [fact_matching_results](fact_matching_results/BEAR-small/wikimedia_wikipedia_20231101_en/evaluation_on_slices/)

same as BEAR-big

#### 1. gpt2_124m
- link to probing results: [probing results](probing_results/BEAR-small/gpt2_124m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-big/gpt2_124m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### 2. xlstm_247m
- link to probing results: [probing results](probing_results/BEAR-small/xlstm_247m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-small/xlstm_247m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### 3. mamba2_172m
- link to probing results: [probing results](probing_results/BEAR-small/mamba2_172m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-small/mamba2_172m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### 4. gpt2_209m
- link to probing results: [probing results](probing_results/BEAR-small/gpt2_209m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-small/gpt2_209m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### 5. gpt2_355m
- link to probing results: [probing results](probing_results/BEAR-small/gpt2_355m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-small/gpt2_355m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### 6. mamba2_432m
- link to probing results: [probing results](probing_results/BEAR-small/mamba2_432m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-small/mamba2_432m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

#### 7. xlstm_406m
- link to probing results: [probing results](probing_results/BEAR-small/xlstm_406m/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](probing_results/BEAR-small/xlstm_406m/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)

### Weighted Accuracy on Checkpoints

Here, the goal is to create a weighted accuracy score for each model checkpoint, boiling down the accuracy on the 
occurrence buckets to a single score.
Here, we proposed two methods:

#### 1. Weighted Accuracy Score on Occurrence Buckets (WASB)

$$\frac{1}{\sum_{i=1}^{N}w_i}\sum_{i=1}^{N}w_i * \text{acc}_i$$

The main idea is
to account for the bucket size change with increasing number of facts per bucket as the model sees more data.
Thus, this method is more suited for a comparison of the checkpoints of one model.

##### Where:

- $w_i$ is the weight of the bucket $i$.
Here, the weights are dependent on the number of occurrences of the a fact 
where each fact is sorted into a bucket (e.g., 0, 2-4, 4-8, ... with the bucked end being exclusive).
- $acc_i$ is the accuracy of the bucket $i$.
- If a fact has an occurrence of 5, the weight is calculated with the occurrence of 4 e.g., 4-8 bucket. (always the lower bound of the bucket).

The weight is thus calculated as follows:

$$
w_i = 
\begin{cases}
exp(-\lambda x), \text{if } x\ge 1\\
0, \text{otherwise}
\end{cases}
$$

- with $\lambda=0.05$
- $x$ is the occurrence of the fact $i$ (the lower bound of the bucket).


- [Results BEAR-big](probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-big/on_buckets/weighted_accuracy_on_slices_bear_big.png)
- [Results BEAR-small](probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-small/on_buckets/weighted_accuracy_on_slices_bear_small.png)

#### 2. Weighted Accuracy Score Over All Facts (WAF)

This method is more suited for a comparison of the different models final checkpoint.

$$\frac{\sum_{i=1}^{N}\widehat{w_i}}{\sum_{i=1}^{N}w_i}$$

##### Where:
The weight is thus calculated as follows:

- $w_i$ is the weight of the fact $i$.
- $\widehat{w_i}$ is the weight of the fact $i$ with the condition that the model answered the fact correctly.

$$
\widehat{w_i} =
\begin{cases}
w_i, \text{if } \widehat{y}=1\\
0, \text{if } \widehat{y}=0\\
\end{cases}
$$

- [Results BEAR-big](probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-big/over_all_facts/weighted_accuracy_on_slices_bear_big.png)
- [Results BEAR-small](probing_results/weighted_accuracy_over_slices/wikimedia_wikipedia_20231101_en/BEAR-small/over_all_facts/weighted_accuracy_on_slices_bear_small.png)


## Correct Answer Probability Analysis

For each checkpoint, derive the probability of the model
to answer a fact correctly given the number of occurrences of the fact in the training data up until the slice
seen by the model at said checkpoint. 

The parameters are optimized
by minimizing the negative log-likelihood of the model's predictions up until the slice seen by the model.

The following probability functions are tested:

#### 1. Cumulative Distribution Function (CDF)

$$F(x; \lambda) = 1 - e^{-\lambda x} , x\ge 0$$

Facts with an occurrence of 0 are set to a probability of 0.

- [Results BEAR-big](correct_answer_probability_analysis_plots/BEAR-big/cumulative_distribution_function)
- [Results BEAR-small](correct_answer_probability_analysis_plots/BEAR-small/cumulative_distribution_function)


#### 2. Power Scaling Function (PSF)

$$F(x; \alpha) = 1 - \left(L_0 + \frac{x_0}{(1+x)^\alpha}\right)$$

- [Results BEAR-big](correct_answer_probability_analysis_plots/BEAR-big/power_scaling_function)
- [Results BEAR-small](correct_answer_probability_analysis_plots/BEAR-small/power_scaling_function)


#### Optimization

$$p_{m,i} = T_{m,i} F(x_i) + (1 - T_{i,m})\left(1 - F(x_i)\right)$$

for $log()$ =>

$$p_{m,i} = T_{m,i}*log(F(x_i)) + (1 - T_{i,m})*log(1 - F(x_i)$$


PSF:

$$P\left(L_0, x_0, \alpha_{m} \right) = \prod_{m \in [M]} \prod_{i \in [N]} p_{m,i}$$

for $log()$ =>

$$P\left(L_0, x_0, \alpha_{m} \right) = \sum_{m \in [M]} \sum_{i \in [N]} p_{m,i}$$


CDF:

$$P\left(\lambda_{m} \right) = \prod_{i \in [N]} p_{m,i}$$

for $log()$ =>

$$P\left(\lambda_{m} \right) = \sum_{i \in [N]} p_{m,i}$$

Where:

#### Where:
 
- $T_i$ is the target value for the fact $i$ (1 if the model answered correctly, 0 otherwise).
- Summation is normalized by -1/num_samples

### Probability Function Analysis

For each model, we compare the negative log-likelihood of the model's predictions up until the slice seen by the model
for each probability function mentioned above.
The function with the lowest negative log-likelihood is chosen as the best fit.
The results are found within each model directory in:
- `probin_results/BEAR-{size}/{model}/wikimedia_wikipedia_20231101_en/evaluation_on_slices/correct_answer_probability_optimized_params`.
