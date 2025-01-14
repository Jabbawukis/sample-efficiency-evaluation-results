# Sample Efficiency Evaluation Results

## For [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)

## Probing on the whole dataset:

Train a model on the whole dataset and probe it after training.

### BEAR-big
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en)

#### Aliases Stats:

- % of subjects with alias: 0.4271537429432166
- % of objects with alias: 0.7703662182361734
- Overall number of instances with matches: 28144
- Number of instances with more matches achieved due to aliases: 13032
- Number of instances with matches achieved without the need for aliases: 15112
- Number of instances with no matches with aliases: 12772
- Number of instances with no matches without aliases: 15875
- % of instances with more matches having subject and object aliases: 0.5860957642725598
- % of instances with more matches having only subject aliases: 0.06269183548189074
- % of instances with more matches having only object aliases: 0.3512124002455494
- % of instances with more matches due to aliases (over all instances with matches): 0.4630471859010802
- Average increase in matches due to aliases: 187.09484346224679

##### 1. gpt2_off_the_shelve (for comparison)

- Model: gpt2 (137M params)
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_off_the_shelve/wikimedia_wikipedia_20231101_en)
- trained on: a pre-trained model

#### lm-evaluation-harness scores
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.5162|±  | 0.014|
|wsc273|       1 |none  |     0|acc   |↑  |0.5861|±  |0.0299|
|lambada_standard|       1 |none  |     0|acc       |↑  | 0.2597|±  |0.0061|
|lambada_standard|       1 |none  |     0|perplexity|↓  |93.7302|±  |3.8329|
|pile_10k|       1 |none  |     0|bits_per_byte  |↓  |  1.1745|±  |   N/A|
|pile_10k|       1 |none  |     0|byte_perplexity|↓  |  2.2572|±  |   N/A|
|pile_10k|       1 |none  |     0|word_perplexity|↓  |233.5492|±  |   N/A|


- Model: gpt2-large (812M params)

|     Tasks      | Version |Filter|n-shot|  Metric  |   | Value |   |Stderr|
|----------------|--------:|------|-----:|----------|---|------:|---|-----:|
|lambada_standard|       1 |none  |     0|acc       |↑  | 0.4040|±  |0.0068|
|lambada_standard|       1 |none  |     0|perplexity|↓  |22.1789|±  |0.7740|

#### 2. gpt2_from_scratch

- Model: gpt2 (124M params)
- repo: [J4bb4wukis/gpt2_wikipedia_en](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en)
- trained on: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

#### lm-evaluation-harness scores
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.4925|±  |0.0141|
|wsc273|       1 |none  |     0|acc   |↑  |0.5531|±  |0.0301|
|lambada_standard|       1 |none  |     0|acc       |↑  |  0.1527|±  | 0.0050|
|lambada_standard|       1 |none  |     0|perplexity|↓  |880.4093|±  |44.9954|
|pile_10k|       1 |none  |     0|bits_per_byte  |↓  |    2.0130|±  |   N/A|
|pile_10k|       1 |none  |     0|byte_perplexity|↓  |    4.0362|±  |   N/A|
|pile_10k|       1 |none  |     0|word_perplexity|↓  |11459.1370|±  |   N/A|

### BEAR(-small)
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-small/wikimedia_wikipedia_20231101_en)

#### Aliases Stats:
- % of subjects with alias: 0.5051875498802874
- % of objects with alias: 0.8207317073170731
- Overall number of instances with matches: 6775
- Number of instances with more matches achieved due to aliases: 3603
- Number of instances with matches achieved without the need for aliases: 3172
- Number of instances with no matches with aliases: 956
- Number of instances with no matches without aliases: 1463
- % of instances with more matches having subject and object aliases: 0.6319733555370525
- % of instances with more matches having only subject aliases: 0.04607271718012767
- % of instances with more matches having only object aliases: 0.32195392728281985
- % of instances with more matches due to aliases (over all instances with matches): 0.5318081180811808
- Average increase in matches due to aliases: 297.6960865945046

#### 1. gpt2_off_the_shelve (for comparison)

- Model: gpt2 (137M params)
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](/probing_results/BEAR-small/gpt2_off_the_shelve/wikimedia_wikipedia_20231101_en)
- trained on: a pre-trained model

#### 2. gpt2_from_scratch

- Model: gpt2 (124M params)
- repo: [J4bb4wukis/gpt2_wikipedia_en](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en)
- link to probing results: [probing results](/probing_results/BEAR-small/gpt2_from_scratch/wikimedia_wikipedia_20231101_en)
- trained on: [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)
- training script: [train.py](https://github.com/Jabbawukis/sample_efficiency_evaluation/blob/main/model_training_setups/GPT2/wikimedia_wikipedia_20231101_en/train.py)

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

#### 1. gpt2_from_scratch
- Model: gpt2 (124M params)
- repo (model checkpoints as branches): [J4bb4wukis/gpt2_wikipedia_en_shuffeld](https://huggingface.co/J4bb4wukis/gpt2_wikipedia_en_shuffeld)
- dataset shuffle seed: 42
- number of slices: 42
- per_device_train_batch_size: 32
- gradient_accumulation_steps: 8
- save_steps: 3650 (per slice num_rows_after_tokenized avg. ≈ 934,840 → 934,840 ÷ 8 ÷ 32 ≈ 3650)
- context_length 128
- link to slice info: [evaluation_on_slices](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/evaluation_on_slices)
- link to probing results: [probing results](/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/)

#### lm-evaluation-harness scores (final model checkpoint)
|  Tasks   | Version |Filter|n-shot|Metric|   |Value |   |Stderr|
|----------|--------:|------|-----:|------|---|-----:|---|-----:|
|winogrande|       1 |none  |     0|acc   |↑  |0.5193|±  | 0.014|
|wsc273|       1 |none  |     0|acc   |↑  |0.5165|±  |0.0303|
|lambada_standard|       1 |none  |     0|acc       |↑  |  0.1558|±  | 0.0051|
|lambada_standard|       1 |none  |     0|perplexity|↓  |822.1627|±  |42.0769|
|pile_10k|       1 |none  |     0|bits_per_byte  |↓  |    2.0200|±  |   N/A|
|pile_10k|       1 |none  |     0|byte_perplexity|↓  |    4.0560|±  |   N/A|
|pile_10k|       1 |none  |     0|word_perplexity|↓  |11840.3982|±  |   N/A|

Other training parameters:
- logging_steps: 3650
- num_train_epochs: 1
- weight_decay: 0.1
- warmup_steps: 1_000
- lr_scheduler_type: "cosine"
- learning_rate: 5e-4
- fp16: True

#### Accuracy on checkpoints

For each checkpoint,
we count the number of facts with specific occurrences up until the slice seen by the model at said checkpoint.
The model checkpoint is probed and the accuracy depending on the number of occurrences up until the slice is calculated.

- link to accuracy diagrams on checkpoints: [accuracy_on_checkpoints](/probing_results/BEAR-big/gpt2_from_scratch/wikimedia_wikipedia_20231101_en/evaluation_on_slices/combined_accuracy_plots_grid.png)