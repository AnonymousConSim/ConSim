# ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability

## Instalation

Becareful, this repository uses both PyTorch and TensorFlow.
```
pip install -e .
```


## Launching experiments

First you need to download datasets and adapt `src/utils/dataset_utils.py` to load your datasets. You will also have to adapt `src/utils/models_configs.py` to create model configs for your dataset. Finally, you might also have to add a prompting relative to the dataset for `SplittedLlamaForCausalLM` in `src/utils/splitted_models.py`.

```
train_evaluate.py --dataset your_dataset --model your_model
llama_embedings.py --dataset your_dataset
compute_concepts_and_co.py --dataset your_dataset --model your_model
concepts_communication.py --dataset your_dataset --model your_model
make_prompts.py --dataset your_dataset --model your_model
call_openai_api.py --dataset your_dataset --model your_model
compute_methods_perf.py
visualize_methods_perfs.py
```
