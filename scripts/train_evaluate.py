"""
Fine-tune a model on a dataset or evaluate the given model.
"""
# built-in imports
import os
import shutil
import time

# libraries imports
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

# local imports
from utils.datasets_utils import load_split_dataset, load_huggingface_dataset_parts
from utils.general_utils import get_args, select_best_gpu, get_free_memory_per_gpu
from utils.models_configs import model_name_from_config, get_splitted_model, create_model
from utils.models_inference import get_all_embeddings, get_logits_from_embeddings
from utils.text_utils import split_paragraph_and_repeat_labels


# Select the best GPU
best_gpu = select_best_gpu()

# Set the selected GPU as the device in PyTorch
DEVICE = torch.device(f'cuda:{best_gpu}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE)
print(f'Selected GPU: {best_gpu} with {get_free_memory_per_gpu()[best_gpu] / 1024**3:.2f} GB free memory')

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("DEVICE:", DEVICE)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

def main():
    # get args
    dataset_config, model_config, args = get_args()
    force_training = args.force
    embeddings_input_granularity = args.granularity

    # print("DEBUG: train_evaluate.py: dataset_config:", dataset_config)
    # print("DEBUG: train_evaluate.py: model_config:", model_config)
    
    tokenizer_batch_size = 1024
    end_model_batch_size = 512

    print(f"\nModel {str(model_config.model_class)} on {dataset_config.name}, " +\
          f"force training to {force_training}.")

    # create model folders if not present
    models_folder = os.path.join(dataset_config.path, "models")
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, model_name_from_config(model_config))
    os.makedirs(model_path, exist_ok=True)

    if force_training:
        # remove previous model
        shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)
    
    # check if a model already exists at model_path
    if ("config.json" in os.listdir(model_path) or "score.pth" in os.listdir(model_path)) and not force_training:
        print(f"\nFound model at {model_path}, loading it...")
        model, tokenizer = get_splitted_model(model_config, dataset_config, device=DEVICE)
    else:
        if not os.listdir(model_path):
            print(f"\nNo model found at {model_path}, training a new one...")
        else:
            print(f"\nForcing training, removing previous model and training a new one...")
            concepts_saves_path = os.path.join(os.getcwd(), "data", "concepts", dataset_config.name, model_name_from_config(model_config))
            shutil.rmtree(concepts_saves_path, ignore_errors=True)
            shutil.rmtree(model_path, ignore_errors=True)

        # load dataset
        train_dataset, test_dataset = load_huggingface_dataset_parts(dataset_config)
        
        train_args = TrainingArguments(
            output_dir=model_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=model_config.learning_rate,
            # lr_scheduler_type="constant_with_warmup",  # maybe I just need this
            num_train_epochs=model_config.num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
            per_device_train_batch_size=model_config.batch_size,
            per_device_eval_batch_size=model_config.batch_size,  # TODO: remove the /2
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            eval_accumulation_steps=1,
            remove_unused_columns=False,
        )

        # initialize model and tokenizer
        model, tokenizer = create_model(model_config, dataset_config.num_labels, DEVICE)
        
        tokenize_function = lambda examples: tokenizer(examples["text"],
                                                       padding="max_length",
                                                       truncation=True,
                                                       max_length=model_config.max_length,
                                                       return_tensors="pt")

        # tokenize datasets
        print("\nMapping train and test dataset with the tokenizer:")
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=tokenizer_batch_size)
        tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=tokenizer_batch_size)
        del train_dataset, test_dataset

        # Custom metric computation function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = np.argmax(logits, axis=1)
            accuracy = (preds == labels).mean()
            return {'accuracy': accuracy}
        
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            compute_metrics=compute_metrics,
            optimizers=(torch.optim.Adam(params=model.parameters(), lr=model_config.learning_rate), None),  # used for non-deberta
        )

        print("\nTraining the model...")
        trainer.train()

        trainer.save_model(model_path)
        print(f"\nModel trained and saved at {model_path}.")
        if "llama" in args.model:
            # save the last dense layer weights
            torch.save(model.score.state_dict(), os.path.join(model_path, "score.pth"))

        del trainer
        del tokenized_train_dataset, tokenized_test_dataset

        # load model and tokenizer
        if not model_config.positive_embeddings:
            del model, tokenizer
            model, tokenizer = get_splitted_model(model_config, dataset_config, device=DEVICE)

    print("\nEvaluating the model on the train and test set...")

    train_data, train_labels, _, _, test_data, test_labels = load_split_dataset(dataset_config)
    if isinstance(train_data, pd.DataFrame):
        train_data = list(train_data["sentence"])
        test_data = list(test_data["sentence"])

    # load embeddings or compute them
    train_embeddings, _, test_embeddings = get_all_embeddings(
        dataset_path=dataset_config.path, model_config=model_config, regenerate=force_training,
        model=model, tokenizer=tokenizer, device=DEVICE, granularity=embeddings_input_granularity,
        train_inputs=train_data, val_inputs=None, test_inputs=test_data)
    
    model.cpu()
    
    if embeddings_input_granularity in ["sentence", "sentence-part"]:
        _, train_labels = split_paragraph_and_repeat_labels(train_data, train_labels, embeddings_input_granularity)
        _, test_labels = split_paragraph_and_repeat_labels(test_data, test_labels, embeddings_input_granularity)
    elif embeddings_input_granularity == "unique-words":
        print("Accuracy makes no sense for unique words. Exiting.")
        exit()

    train_preds = get_logits_from_embeddings(model, train_embeddings,
                                             batch_size=end_model_batch_size, device=DEVICE)
    test_preds = get_logits_from_embeddings(model, test_embeddings,
                                            batch_size=end_model_batch_size, device=DEVICE)

    train_accuracy = accuracy_score(train_preds.argmax(axis=-1), train_labels)
    test_accuracy = accuracy_score(test_preds.argmax(axis=-1), test_labels)
    print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
