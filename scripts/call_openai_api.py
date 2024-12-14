import json
import os
import shutil

from utils.general_utils import get_args
from utils.models_configs import model_name_from_config

import openai
import tiktoken


costs = {
    "gpt-3.5-turbo": {"input_tokens": 0.5 / 1000000, "output_tokens": 1.5 / 1000000},
    "gpt-4o": {"input_tokens": 5 / 1000000, "output_tokens": 15 / 1000000},
    "gpt-4o-mini": {"input_tokens": 0.15 / 1000000, "output_tokens": 0.6 / 1000000},
}


def compute_and_show_costs(open_ai_model, nb_calls, nb_input_tokens, nb_output_tokens):

    input_cost = nb_input_tokens * costs[open_ai_model]["input_tokens"]
    output_cost = nb_output_tokens * costs[open_ai_model]["output_tokens"]
    total_cost = input_cost + output_cost

    print(f"\tEffective costs: input_cost: {round(input_cost, 4)}$,",
          f"output_cost: {round(output_cost, 4)}$,",
          f"total_cost: {round(total_cost, 4)}$.")

    print(f"\tMean costs: input_cost: {round(input_cost / nb_calls, 4)}$,",
          f"output_cost: {round(output_cost / nb_calls, 4)}$,",
          f"total_cost: {round(total_cost / nb_calls, 4)}$.")


def main():
    #arguments
    dataset_config, model_config, args = get_args()
    force_regeneration = args.force
    open_ai_model = "gpt-4o-mini"  # "gpt-3.5-turbo"  # TODO: change to gpt-4o
    model_name = model_name_from_config(model_config)
    seeds = [0, 1, 2, 3, 4, 5, 6]
    nb_correct =20  # TODO: increase and compare results
    nb_mistakes = 20  # TODO: increase and compare results
    samples_setting = f"{nb_correct}-{nb_mistakes}samples"
    concepts_communication_suffixes = ["concept_activating_words", "o1_concepts_correspondence"]

    print(f"LLM {open_ai_model} analyzing prompts for:",
          f"model: {model_name},",
          f"dataset: {dataset_config.name},",
          f"force recomputing: {force_regeneration}.")

    # load api key
    open_ai_api_key = open("open_ai_key.txt", "r").read().strip()

    client = openai.OpenAI(
        api_key=open_ai_api_key,
        organization=...,
        project=...,
    )
    tokenizer = tiktoken.encoding_for_model(open_ai_model)

    data_keys = ["labels", "f_predictions", "fc_predictions"]

    nb_calls = 0
    nb_input_tokens = 0
    nb_output_tokens = 0

    for seed in seeds:

        prompts_seed_dir = os.path.join(os.getcwd(), "data", "prompts", samples_setting, dataset_config.name, model_name, f"seed{seed}")
        if not os.path.exists(prompts_seed_dir) and not os.listdir(prompts_seed_dir):
            raise ValueError(f"\nNo prompts found at path {prompts_seed_dir}.")

        responses_path = os.path.join(os.getcwd(), "data", "responses", open_ai_model, samples_setting, dataset_config.name, model_name, f"seed{seed}")
        print(f"Responses will be saved at: {responses_path}")
        os.makedirs(responses_path, exist_ok=True)

        lower_bound_responses = {}

        if force_regeneration:
            shutil.rmtree(responses_path, ignore_errors=True)
        else:
            if os.path.exists(responses_path) and (len(os.listdir(responses_path)) == len(os.listdir(prompts_seed_dir))):
                print(f"\nAll responses already computed, skipping.")
                continue

            for prompts_file_name in os.listdir(prompts_seed_dir):
                if prompts_file_name == "sentences_indices.npy":
                    continue

                print(f"\nPrompt path: {prompts_seed_dir}")
                print(f"Prompt file: {prompts_file_name}")

                prompts_path = os.path.join(prompts_seed_dir, prompts_file_name)

                with open(prompts_path, 'r') as json_file:
                    prompts = json.load(json_file)

                # checks if responses already exist
                response_path = os.path.join(responses_path, prompts_file_name)
                os.makedirs(os.path.dirname(response_path), exist_ok=True)
                if os.path.exists(response_path) and not force_regeneration:
                    with open(response_path, 'r') as json_file:
                        responses = json.load(json_file)

                    if len(responses) == len(prompts):
                        if len(lower_bound_responses) < 4:
                            lower_bound_responses = {
                                xp_name: response
                                for xp_name, response in responses.items()
                                if xp_name[0] == "L"
                            }
                        print(f"\tAll prompts already answered, skipping.")
                        continue
                else:
                    responses = {}

                if len(prompts) == 0:
                    print(f"\tNo prompts found, artefact of failed experiment.")
                    continue
                else:
                    local_calls = 0
                    local_input_tokens = 0
                    local_output_tokens = 0
                    for prompt_type, prompt in prompts.items():

                        if prompt_type in data_keys:
                            # this are not prompts but data
                            continue

                        # skip prompts for already answered questions
                        if prompt_type in responses.keys() or prompt_type in lower_bound_responses.keys():
                            print(f"\t{prompt_type} already answered, skipping.")
                            continue

                        nb_tokens = len(tokenizer.encode(str(prompt)))
                        if nb_tokens > 16360 and open_ai_model == "gpt-3.5-turbo":
                            print(f"\tPrompt too long, {nb_tokens} tokens, skipping.")
                            continue

                        if nb_tokens > 128_000 and open_ai_model == "gpt-4o-mini":
                            print(f"\tPrompt too long, {nb_tokens} tokens, skipping.")
                            continue
                        # print prompt
                        # print("\n\n\n\nPrompt type:", prompt_type)
                        # for prompt_part in prompt:
                        #     print("\n\nRole:", prompt_part["role"])
                        #     print("Content:\n", prompt_part["content"])

                        print(f"\tCalling {open_ai_model} for prompt: {prompt_type}")
                        
                        # for baselines call the model 5 times with different seeds
                        response = client.chat.completions.create(
                            model=open_ai_model,
                            seed=0,
                            messages=prompt,
                            temperature=0,
                        )

                        local_calls += 1
                        local_input_tokens += response.usage.prompt_tokens
                        local_output_tokens += response.usage.completion_tokens

                        # extracting response content
                        responses[prompt_type] = response.choices[0].message.content
                    
                    if len(lower_bound_responses) < 4:
                        lower_bound_responses = {
                            xp_name: response
                            for xp_name, response in responses.items()
                            if xp_name[0] == "L"
                        }
                    
                    responses.update(lower_bound_responses)
                    responses.update({data_key: prompts[data_key]
                                      for data_key in data_keys})
                
                    if local_calls > 0:
                        compute_and_show_costs(open_ai_model, local_calls, local_input_tokens, local_output_tokens)
                        nb_calls += local_calls
                        nb_input_tokens += local_input_tokens
                        nb_output_tokens += local_output_tokens
                
                with open(response_path, 'w') as json_file:
                    json.dump(responses, json_file, indent=4)

        if nb_calls > 0:
            print("\n\nGlabal costs:")
            print(f"model: {open_ai_model}, nb_calls: {nb_calls}, nb_input_tokens: {nb_input_tokens}, nb_output_tokens: {nb_output_tokens}")
            compute_and_show_costs(open_ai_model, nb_calls, nb_input_tokens, nb_output_tokens)


if __name__ == "__main__":
    main()
