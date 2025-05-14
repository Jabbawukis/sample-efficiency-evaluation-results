## For [wikipedia_20231101_en](https://huggingface.co/datasets/wikimedia/wikipedia)

## Probing on the whole dataset:

Train a model on the whole dataset and probe it after training.

## BEAR-big
- fact matching results: [fact_matching_results](fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en)

### Aliases Stats

- % of subjects with alias: 0.4271537429432166
- % of objects with alias: 0.7703662182361734
- Overall number of instances with matches: 28061
- Number of instances with more matches achieved due to aliases: 12675
- Number of instances with matches achieved without the need for aliases: 15386
- Number of instances with no matches with aliases: 12855
- Number of instances with no matches without aliases: 15917
- % of instances with more matches having subject and object aliases: 0.5917948717948718
- % of instances with more matches having only subject aliases: 0.06390532544378698
- % of instances with more matches having only object aliases: 0.3442998027613412
- % of instances with more matches due to aliases (over all instances with matches): 0.45169452264709026
- Average increase in matches (per fact) due to aliases: 55.69720891582755

### 1. gpt2_137m_off_the_shelve (for comparison)

- Model: GPT2 (137M params)
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](probing_results/BEAR-big/gpt2_137m_off_the_shelve/wikimedia_wikipedia_20231101_en/accuracy_statistics.png)
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


- Model: gpt2-medium (380M params)

|     Tasks      | Version |Filter|n-shot|  Metric  |   | Value |   |Stderr|
|----------------|--------:|------|-----:|----------|---|------:|---|-----:|
|winogrande|      1|none  |     0|acc   |↑  |0.5312|±  | 0.014|
|wsc273|      1|none  |     0|acc   |↑  |0.6117|±  |0.0296|
|lambada_standard|      1|none  |     0|acc       |↑  | 0.3769|±  |0.0068|
|lambada_standard|      1|none  |     0|perplexity|↓  |29.6928|±  |1.0742|
|pile_10k| 1|none  |     0|bits_per_byte  |↓  |  1.0493|±  |   N/A|
|pile_10k| 1|none  |     0|byte_perplexity|↓  |  2.0695|±  |   N/A|
|pile_10k| 1|none  |     0|word_perplexity|↓  |130.5904|±  |   N/A|

### 2. gpt2_124m

see for model details: [gpt2_124m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-big/gpt2_124m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_gpt2_124m_bear_big.png)

### 3. xlstm_247m

see for model details: [xlstm_247m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-big/xlstm_247m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_xlstm_247m_bear_big.png)

### 4. mamba2_172m

see for model details: [mamba2_172m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-big/mamba2_172m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_mamba2_172m_bear_big.png)

### 5. gpt2_209m

see for model details: [gpt2_209m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-big/gpt2_209m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_gpt2_209m_bear_big.png)

### 6. gpt2_355m

see for model details: [gpt2_355m](probing_on_dataset_slices.md)

### 7. mamba2_432m

see for model details: [mamba2_432m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-big/mamba2_432m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_mamba2_432m_bear_big.png)

### 8. xlstm_406m

see for model details: [xlstm_406m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-big/xlstm_406m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_xlstm_406m_bear_big.png)

### 9. llama_360m
see for model details: [llama_360m](probing_on_dataset_slices.md)
- link to probing results (final model): [probing results](probing_results/BEAR-big/llama_360m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_llama_360m_bear_big.png)

### 10. llama_208m
see for model details: [llama_208m](probing_on_dataset_slices.md)
- link to probing results (final model): [probing results](probing_results/BEAR-big/llama_208m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_llama_208m_bear_big.png)

## BEAR(-small)
- fact matching results: [fact_matching_results](/fact_matching_results/BEAR-small/wikimedia_wikipedia_20231101_en)

#### Aliases Stats
- % of subjects with alias: 0.5051875498802874
- % of objects with alias: 0.8207317073170731
- Overall number of instances with matches: 6765
- Number of instances with more matches achieved due to aliases: 3503
- Number of instances with matches achieved without the need for aliases: 3262
- Number of instances with no matches with aliases: 966
- Number of instances with no matches without aliases: 1461
- % of instances with more matches having subject and object aliases: 0.6383100199828718
- % of instances with more matches having only subject aliases: 0.04738795318298601
- % of instances with more matches having only object aliases: 0.31430202683414216
- % of instances with more matches due to aliases (over all instances with matches): 0.5178122690317812
- Average increase in matches (per fact) due to aliases: 131.71362048894062

### 1. gpt2_137m_off_the_shelve (for comparison)

- Model: GPT2 (137M params)
- repo: [openai-community/gpt2](https://huggingface.co/gpt2)
- link to probing results: [probing results](probing_results/BEAR-small/gpt2_137m_off_the_shelve/wikimedia_wikipedia_20231101_en/accuracy_statistics.png)
- trained on: a pre-trained model

### 2. gpt2_124m

see for model details: [gpt2_124m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/gpt2_124m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_gpt2_124m_bear_small.png)

### 3. xlstm_247m

see for model details: [xlstm_247m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/xlstm_247m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_xlstm_247m_bear_small.png)

### 4. mamba2_172m

see for model details: [mamba2_172m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/mamba2_172m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_mamba2_172m_bear_small.png)

### 5. gpt2_209m

see for model details: [gpt2_209m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/gpt2_209m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_gpt2_209m_bear_small.png)

### 6. gpt2_355m

see for model details: [gpt2_355m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/gpt2_355m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_gpt2_355m_bear_small.png)

### 7. mamba2_432m

see for model details: [mamba2_432m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/mamba2_432m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_mamba2_432m_bear_small.png)

### 8. xlstm_406m

see for model details: [xlstm_406m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/xlstm_406m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_xlstm_406m_bear_small.png)

### 9. llama_360m
see for model details: [llama_360m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/llama_360m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_llama_360m_bear_small.png)

### 10. llama_208m
see for model details: [llama_208m](probing_on_dataset_slices.md)

- link to probing results (final model): [probing results](probing_results/BEAR-small/llama_208m/wikimedia_wikipedia_20231101_en/accuracy_statistics_final_llama_208m_bear_small.png)