import os
from itertools import product, permutations

import pandas as pd

from plot import plot_accuracies_summaries, plot_pairwise_comparison_matrix, plot_prompt_type_pairwise_comparison_matrix


def combinations_iterator(df, baseline_df, verbose=False):
    # datasets
    unique_datasets = sorted(list(df.index.get_level_values('dataset').unique()))
    datasets = {"all": unique_datasets}
    datasets["red"] = ["BIOS10", "tweet_eval_emotion"]
    # datasets.update({dataset: [dataset] for dataset in unique_datasets})

    # models
    unique_models = sorted(list(df.index.get_level_values('model').unique()))
    models = {"all": unique_models}
    # models.update({model: [model] for model in unique_models})
    models["red"] = [model for model in unique_models if model[-1] == "+"]

    # communication
    unique_communications = sorted(list(df.index.get_level_values('communication').unique()))
    communications = {"all": unique_communications}
    # communications.update({communication: [communication] for communication in unique_communications})

    # methods
    unique_methods = sorted(list(df.index.get_level_values('method').unique()))
    methods = {"all": unique_methods}
    # methods.update({method: [method] for method in unique_methods})
    methods["top3"] = ["ICA", "NMF", "SAE"]

    # prompt_types
    unique_prompt_types = list(df.index.get_level_values('prompt_type').unique())
    prompt_types = {"all": unique_prompt_types}
    #prompt_types.update({prompt_type: [prompt_type] for prompt_type in unique_prompt_types})
    # we removed upper bounds!
    prompt_types["all"] = ["E1", "E2", "E3", "E1-a", "E2-a", "E3-a"]  # [prompt_type for prompt_type in unique_prompt_types if prompt_type[0] == "E"]
    # prompt_types["Ex"] = [prompt_type for prompt_type in prompt_types["all"] if len(prompt_type) <= 2]
    # prompt_types["Ex-a"] = [prompt_type for prompt_type in prompt_types["all"] if len(prompt_type) > 2]
    prompt_types["red"] = ["E1", "E2", "E1-a", "E2-a"]
    if prompt_types["all"] == prompt_types["red"]:
        prompt_types.pop("red")

    n_concept = sorted(list(df.index.get_level_values('n_concepts').unique()))
    for (d_name, dataset), (m_name, model), (c_name, communication), (p_name, prompt_type), (mt_name, method) in\
            product(datasets.items(), models.items(), communications.items(), prompt_types.items(), methods.items()):
        
        if sum([d_name == "all", m_name == "all", c_name == "all", p_name == "all", mt_name == "all"]) >= 3\
                or (d_name == "red" and m_name == "red" and c_name == "all" and p_name == "red" and mt_name == "all"):
            path_dirs = [
                f"{index}/{name}"
                for index, name in [
                    ("datasets", d_name),
                    ("models", m_name),
                    ("methods", mt_name),
                    ("communications", c_name),
                    ("prompt_types", p_name),
                ]
                if name != "all"
            ]
            
            try:
                filtered_df = df.loc[pd.IndexSlice[dataset, model, communication, method, n_concept, prompt_type]].dropna().droplevel("n_concepts")
                filtered_baseline_df = baseline_df.loc[pd.IndexSlice[dataset, model, communication, method, n_concept, prompt_type]].dropna().droplevel("n_concepts")
            except KeyError:
                continue

            if verbose:
                print(f"Filtering dataset, plots dir: {path_dirs}")
            
            if c_name == "all":
                yield filtered_df, filtered_baseline_df, path_dirs, "communication"
            
            if mt_name == "all":
                yield filtered_df, filtered_baseline_df, path_dirs, "method"
            
            if p_name in ["all", "Ex", "Ex-a", "Er"]:
                yield filtered_df, filtered_baseline_df, path_dirs, "prompt_type"


def main():
    nb_correct = 20
    nb_mistakes = 20
    samples_setting = f"{nb_correct}-{nb_mistakes}samples"
    user_llm = "gpt-4o-mini"  # ["gemini-1.5-flash", "gemini-1.5-pro", "gpt-4o-mini"]

    accuracy_col = "f_accuracy"
    test_seeds = [f"seed{i}" for i in range(5)]
    optimal_n_concepts = True

    # load results from csv
    responses_path = os.path.join(os.getcwd(), "data", "responses", user_llm, samples_setting)
    results_path = os.path.join(responses_path, "results.csv")
    
    if not os.path.exists(results_path):
        raise ValueError(f"\nNo results found at path {results_path}.")
    
    results_df = pd.read_csv(results_path)
    results_df["prompt_type"] = results_df["prompt_type"].apply(lambda idx: idx.split(":")[0])

    # make aggregation over seeds
    comparison_columns = ["dataset", "model", "communication", "method", "n_concepts", "prompt_type", "seed"]
    results_df = results_df[results_df["seed"].isin(test_seeds)]
    results_df = results_df.set_index(comparison_columns)[accuracy_col]
    results_df = results_df.groupby(level=comparison_columns).mean() * 100

    if optimal_n_concepts:
        # take the subset where n_concepts is optimal
        best_n_concepts_path = os.path.join(os.getcwd(), "data", "responses", "gpt-4o-mini", samples_setting, "best_val_n_concepts.csv")
        best_val_n_concepts = pd.read_csv(best_n_concepts_path, index_col=["dataset", "method"])[accuracy_col]
        results_reset = results_df.reset_index()
        merged = results_reset.merge(
            best_val_n_concepts.reset_index(),
            left_on=["dataset", "method"],
            right_on=["dataset", "method"],
            suffixes=('', '_best_n_concepts')
        )
        filtered_results = merged[merged["n_concepts"] == merged[accuracy_col + "_best_n_concepts"]]
        results_df = filtered_results.set_index(results_df.index.names)[accuracy_col]

    # extract baselines
    # prompt_types_baselines = {
    #     "E1: concepts without LR": "L1: no LR baseline",
    #     "E2: concepts with LR": "L2: with LR baseline",
    #     "E3: concepts with contributions at LR": "L2: with LR baseline",
    #     "U1: concepts with contributions at LR and inf": "L2: with LR baseline",
    #     "E1-a: concepts without LR": "L1-a: no LR baseline",
    #     "E2-a: concepts with LR": "L2-a: with LR baseline",
    #     "E3-a: concepts with contributions at LR": "L2-a: with LR baseline",
    #     "U1-a: concepts with contributions at LR and inf": "L2-a: with LR baseline",
    # }
    prompt_types_baselines = {
        "E1": "L1",
        "E2": "L2",
        "E3": "L2",
        "U1": "L2",
        "E1-a": "L1-a",
        "E2-a": "L2-a",
        "E3-a": "L2-a",
        "U1-a": "L2-a",
    }

    # find rows to replace and map index
    to_replace = results_df.index.get_level_values('prompt_type').isin(prompt_types_baselines.keys())
    mapped_index = results_df.index.to_frame(index=False)
    mapped_index.loc[to_replace, "prompt_type"] = mapped_index.loc[to_replace, "prompt_type"].map(prompt_types_baselines)

    # create baseline df
    baseline_df = results_df.copy()
    baseline_df.values[to_replace] = results_df.reindex(pd.MultiIndex.from_frame(mapped_index)).values[to_replace]

    # filter dfs
    baseline_df = baseline_df.loc[to_replace]
    results_df = results_df.loc[to_replace]

    # accuracies_keys = results_df.columns

    initial_plots_path = os.path.join(responses_path, "plots")
    os.makedirs(initial_plots_path, exist_ok=True)

    # make function to iterate on filtered df with save path
    for filtered_df, filtered_baseline_df, path_dirs, compared_index in combinations_iterator(results_df, baseline_df, verbose=True):
        if len(filtered_df) == 0:
            continue
        save_dir = os.path.join(initial_plots_path, accuracy_col, *path_dirs)

        if compared_index == "prompt_type":
            plot_prompt_type_pairwise_comparison_matrix(
                accuracies=filtered_df,
                baselines=filtered_baseline_df,
                save_dir=save_dir,
            )
        else:
            plot_pairwise_comparison_matrix(
                accuracies=filtered_df,
                baselines=filtered_baseline_df,
                compared_index=compared_index,
                save_dir=save_dir,
            )

        if compared_index != "prompt_type":
            plot_accuracies_summaries(
                accuracies=filtered_df,
                baselines=filtered_baseline_df,
                compared_index1=compared_index,
                save_dir=save_dir,
            )


if __name__ == "__main__":
    main()