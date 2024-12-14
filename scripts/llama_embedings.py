"""
Fine-tune a model on a dataset or evaluate the given model.
"""
# built-in imports
import os
from tqdm import tqdm

# libraries imports
from sklearn.metrics import accuracy_score
import torch

# local imports
from utils.datasets_utils import load_split_dataset
from utils.general_utils import get_args
from utils.models_configs import get_splitted_model
from utils.text_utils import split_paragraph_and_repeat_labels, count_unique_words



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'


def batch_end_model(model, embeddings, batch_size=None, device="cuda"):
    if batch_size is None:
        embeddings = embeddings.to(device)
        logits = model.end_model(embeddings).cpu()
        embeddings = embeddings.cpu()
    else:
        logits = []
        for i in tqdm(range(0, len(embeddings), batch_size)):
            embeddings_batch = embeddings[i:i+batch_size].to(device)
            batch_logits = model.end_model(embeddings_batch).cpu()
            logits.append(batch_logits)
        embeddings = embeddings.cpu()
        logits = torch.cat(logits, dim=0)
    return logits


def main():
    # get args
    dataset_config, model_config, args = get_args()
    force_recomputing = args.force
    embeddings_input_granularity = args.granularity

    end_model_batch_size = 2048 * 8

    assert args.model == "llama", "This script is only for the llama model."
    assert not args.positive, "llama does not support specifying positive embeddings."

    print(f"\nModel llama on {dataset_config.name}, force recomputing to {force_recomputing}.")

    # create model folders if not present
    embeddings_path = os.path.join(dataset_config.path, "models", "llama", "embeddings")
    os.makedirs(embeddings_path, exist_ok=True)
    
    # load embeddings or compute them
    train_embeddings_path = os.path.join(embeddings_path, "train_embeddings.pt")
    val_embeddings_path = os.path.join(embeddings_path, "val_embeddings.pt")
    test_embeddings_path = os.path.join(embeddings_path, "test_embeddings.pt")

    if embeddings_input_granularity in ["sentence", "sentence-part", "unique-words"]:
        train_embeddings_path = train_embeddings_path.replace(".pt", f"_{embeddings_input_granularity}.pt")
        val_embeddings_path = val_embeddings_path.replace(".pt", f"_{embeddings_input_granularity}.pt")
        test_embeddings_path = test_embeddings_path.replace(".pt", f"_{embeddings_input_granularity}.pt")
    
    with torch.no_grad():
        # load llama model
        if force_recomputing\
                or not os.path.exists(train_embeddings_path)\
                or not os.path.exists(val_embeddings_path)\
                or not os.path.exists(test_embeddings_path):
            model, _ = get_splitted_model(model_config, dataset_config, device=device)
            print(f"Model loaded from {embeddings_path} on {device}.")
        else:
            model, _ = get_splitted_model(model_config, dataset_config, device="cpu")
            print(f"Model loaded from {embeddings_path} on cpu.")
        model.eval()
        model.tqdm = True
        
        print("\nEvaluating the model on the train and test set...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels = load_split_dataset(dataset_config)

        if dataset_config.name[:4] == "BIOS":
            train_data = list(train_data["sentence"])
            val_data = list(val_data["sentence"])
            test_data = list(test_data["sentence"])

        if embeddings_input_granularity in ["sentence", "sentence-part"]:
            train_data, train_labels = split_paragraph_and_repeat_labels(train_data, train_labels, embeddings_input_granularity)
            val_data, val_labels = split_paragraph_and_repeat_labels(val_data, val_labels, embeddings_input_granularity)
            test_data, test_labels = split_paragraph_and_repeat_labels(test_data, test_labels, embeddings_input_granularity)
        elif embeddings_input_granularity == "unique-words":
            train_data = count_unique_words(train_data, ratio_min_threshold=0.0005, save_dir=dataset_config.path)
            val_data = count_unique_words(val_data, ratio_min_threshold=0.0005, save_dir=dataset_config.path)
            test_data = count_unique_words(test_data, ratio_min_threshold=0.0005, save_dir=dataset_config.path)

        if os.path.exists(train_embeddings_path) and not force_recomputing:
            train_embeddings = torch.load(train_embeddings_path)
            print(f"Train embeddings loaded from {train_embeddings_path}.")
        else:
            train_embeddings = model.features(train_data)
            torch.save(train_embeddings, train_embeddings_path)
            print(f"Train embeddings computed and saved at {train_embeddings_path}.")

        if os.path.exists(val_embeddings_path) and not force_recomputing:
            val_embeddings = torch.load(val_embeddings_path)
            print(f"Val embeddings loaded from {val_embeddings_path}.")
        else:
            val_embeddings = model.features(val_data)
            torch.save(val_embeddings, val_embeddings_path)
            print(f"Val embeddings computed and saved at {val_embeddings_path}.")

        if os.path.exists(test_embeddings_path) and not force_recomputing:
            test_embeddings = torch.load(test_embeddings_path)
            print(f"Test embeddings loaded from {test_embeddings_path}.")
        else:
            test_embeddings = model.features(test_data)
            torch.save(test_embeddings, test_embeddings_path)
            print(f"Test embeddings computed and saved at {test_embeddings_path}.")
        
        if embeddings_input_granularity == "unique-words":
            print("Accuracy makes no sense for unique words. Exiting.")
            exit()
        
        model.to("cpu")
        model.end_model_to(device)
        model.end_model_to(torch.float32)

        # get predictions
        train_preds = batch_end_model(model, train_embeddings.to(torch.float32),
                                      end_model_batch_size, device).numpy()   
        val_preds = batch_end_model(model, val_embeddings.to(torch.float32),
                                    end_model_batch_size, device).numpy()        
        test_preds = batch_end_model(model, test_embeddings.to(torch.float32),
                                     end_model_batch_size, device).numpy()

        # compute accuracies
        train_accuracy = accuracy_score(train_preds.argmax(axis=-1), train_labels)
        val_accuracy = accuracy_score(val_preds.argmax(axis=-1), val_labels)
        test_accuracy = accuracy_score(test_preds.argmax(axis=-1), test_labels)
        print(f"Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}, Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
