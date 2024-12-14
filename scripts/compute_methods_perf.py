import json
import os
from typing import List, Optional

import pandas as pd

from utils.datasets_utils import __configs as dataset_configs


def extract_predictions_from_response(response: str, expected_length: List[int] = None):
    sentences = response.split("\n")

    while sentences[-1] == "":
        sentences = sentences[:-1]

    if expected_length is not None and len(sentences) not in expected_length:
        return []

    # format not respected, expecting predictions to be only separated by "\n"
    if ":" not in sentences[0]:
        return sentences
    
    # expected format: Sample_0: physician\nSample_1: surgeon\nSample_2: nurse
    predictions = [
        sentence.split(": ")[1].strip().lower().split(" ")[0]
        for sentence in sentences
        if (sentence[:10] == "Prediction" or sentence[:8] == "Sentence" or sentence[:6] == "Sample")
    ]
    return predictions


def predictions_accuracy(predictions1: list, predictions2: list, masking: Optional[List[bool]] = None):
    if len(predictions1) == 0 or len(predictions2) == 0:
        print(f"\tNo predictions found. {len(predictions1)} predictions1, {len(predictions2)} predictions2.")
        return None

    if len(predictions1) != len(predictions2) and len(predictions1):
        raise ValueError(f"Predictions lists have different lengths: {len(predictions1)} and {len(predictions2)}.")
    
    if masking is None:
        n_correct = sum([1 for pred1, pred2 in zip(predictions1, predictions2) if pred1 == pred2])

        # assuming corrupted predictions
        # if n_correct == 0:
        #     print("\tNo correct predictions. Assuming corrupted predictions.")
        #     return None

        return round(n_correct / len(predictions1), 3)

    # count correct predictions with initial correct and incorrect predictions
    n_correct_true = sum([1 for pred1, pred2, mask in zip(predictions1, predictions2, masking) if pred1 == pred2 and mask])
    n_correct_false = sum([1 for pred1, pred2, mask in zip(predictions1, predictions2, masking) if pred1 == pred2 and not mask])
    n_correct = n_correct_true + n_correct_false

    # if n_correct == 0:
    #     print("\tNo correct predictions. Assuming corrupted predictions.")
    #     return None
    
    global_acc = round(n_correct / len(predictions1), 3)
    true_acc = round(n_correct_true / sum(masking), 3) if sum(masking) != 0 else None
    false_acc = round(n_correct_false / (len(predictions1) - sum(masking)), 3) if sum(masking) != len(predictions1) else None

    return global_acc, true_acc, false_acc


def main():
    print("\tForcing activating words from unique words.")
    granularity = "unique-words"
    nb_correct = 20
    nb_mistakes = 20
    samples_setting = f"{nb_correct}-{nb_mistakes}samples"
    user_llm = "gpt-4o-mini"  # ["gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o-mini"]

    evaluation_path = os.path.join(os.getcwd(), "data", "responses", user_llm, samples_setting)
    os.makedirs(evaluation_path, exist_ok=True)

    responses_path = os.path.join(os.getcwd(), "data", "responses", user_llm, samples_setting)
    if not os.path.exists(responses_path) and not os.listdir(responses_path):
        raise ValueError(f"\nNo responses found at path {responses_path}.")

    results_list = []
    for dataset_name in os.listdir(responses_path):
        responses_dataset_path = os.path.join(responses_path, dataset_name)
        if os.path.isfile(responses_dataset_path):
            continue
        if "plots" in dataset_name or os.path.isfile(responses_dataset_path):
            continue
        classes_names = dataset_configs[dataset_name].labels_names
        for model_id in os.listdir(responses_dataset_path):
            responses_model_path = os.path.join(responses_dataset_path, model_id)
            model_name = model_id.split("-")[0]
            if model_id.split("_")[-1] == "positive":
                model_name += "+"
            for seed in os.listdir(responses_model_path):
                responses_seed_path = os.path.join(responses_model_path, seed)
                for file_name in os.listdir(responses_seed_path):
                    method_name = "_".join(file_name.split("concepts")[0].split("_")[:-1])
                    n_concepts = int(file_name.split("concepts")[0].split("_")[-1])
                    communication = "CMAW" if "activating" in file_name else "o1CA"

                    print(f"Analyzing {dataset_name} {model_name} {seed} communication:{communication} {method_name} {n_concepts} concepts.")
                    
                    results = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "method": method_name,
                        "n_concepts": n_concepts,
                        "communication": communication
                    }

                    with open(os.path.join(responses_seed_path, file_name), 'r') as json_file:
                        responses = json.load(json_file)

                    if len(responses) == 0:
                        print(f"\tNo responses found in {os.path.join(responses_seed_path, file_name)}, artefact of failed experiments.")
                        results["prompt_type"] = None
                        results["accuracy"] = None
                        results_list.append(results.copy())
                        # print("\tNo response:", results)
                        continue

                    nb_initial_preds = nb_correct + nb_mistakes
                    predictions = {
                        prompt_type: extract_predictions_from_response(response, expected_length=[nb_initial_preds // 2, nb_initial_preds])
                        for prompt_type, response in responses.items()
                    }

                    # convert anonymous classes to initial classes
                    predictions.update({
                        prompt_type: [
                            classes_names[int(pred[-1])]
                            if pred != "none" and int(pred[-1]) < len(classes_names) else classes_names[-1]
                            for pred in prediction
                        ]
                        for prompt_type, prediction in predictions.items()
                        if prompt_type[2:5] == "-a:" if len(prediction) > 0
                    })

                    # small post process to remove mistakes
                    for prompt_type in predictions.keys():
                        if len(predictions[prompt_type]) == nb_initial_preds:
                            # correcting predictions where the GPT predicts for all samples
                            predictions[prompt_type] = predictions[prompt_type][nb_initial_preds//2:]

                    # compute f-accuracies
                    f_masking = [
                        label == pred
                        for label, pred in zip(predictions["labels"], predictions["f_predictions"])
                    ]

                    f_accuracies = {
                        prompt_type: predictions_accuracy(predictions[prompt_type], predictions["f_predictions"], f_masking)
                        for prompt_type in predictions.keys()
                        if "predictions" not in prompt_type and prompt_type != "labels"
                    }

                    # compute fc-accuracies
                    fc_masking = [
                        label == pred
                        for label, pred in zip(predictions["labels"], predictions["fc_predictions"])
                    ]
                    fc_accuracies = {
                        prompt_type: predictions_accuracy(predictions[prompt_type], predictions["fc_predictions"], fc_masking)
                        for prompt_type in predictions.keys()
                        if "predictions" not in prompt_type and prompt_type != "labels"
                    }

                    f_fc_accuracy = predictions_accuracy(predictions["f_predictions"], predictions["fc_predictions"])

                    for prompt_type in f_accuracies.keys():
                        results["prompt_type"] = prompt_type
                        results["f_accuracy"] = f_accuracies[prompt_type][0] if f_accuracies[prompt_type] is not None else None
                        results["f_true_accuracy"] = f_accuracies[prompt_type][1] if f_accuracies[prompt_type] is not None else None
                        results["f_false_accuracy"] = f_accuracies[prompt_type][2] if f_accuracies[prompt_type] is not None else None
                        results["fc_accuracy"] = fc_accuracies[prompt_type][0] if fc_accuracies[prompt_type] is not None else None
                        results["fc_true_accuracy"] = fc_accuracies[prompt_type][1] if fc_accuracies[prompt_type] is not None else None
                        results["fc_false_accuracy"] = fc_accuracies[prompt_type][2] if fc_accuracies[prompt_type] is not None else None
                        results["f_fc_accuracy"] = f_fc_accuracy
                        results_list.append(results.copy())

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(evaluation_path, "results.csv"), index=False)

if __name__ == "__main__":
    main()
