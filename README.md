# ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability

## Instalation

Becareful, this repository uses both PyTorch and TensorFlow.
```
pip install -e .
```


## Launching experiments

First you need to download datasets and adapt `src/utils/dataset_utils.py` to load your datasets. You will also have to adapt `src/utils/models_configs.py` to create model configs for your dataset. Finally, you might also have to add a prompting relative to the dataset for `SplittedLlamaForCausalLM` in `src/utils/splitted_models.py`.

