# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import torch
import time
from typing import Optional, Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE


def get_results_file_path(model_type: str, model_variant: str, num_examples: int,
                          experiment_id: str = "", batched: bool = True) -> str:
    return os.path.join(main_experiment_results_dir(experiment_id),
                        f"{model_type}_{model_variant}_batched_{batched}_num_ex_{num_examples}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                  task_name: str, num_examples: int, batched: bool) -> Tuple[dict, dict]:
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    num_test_datasets, num_dev_datasets = 400, 100
    # num_test_datasets, num_dev_datasets = 75, 0
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    icl_predictions = run_icl(model, tokenizer, task, test_datasets, batched=batched)
    print('ICL predictions', icl_predictions[:10])
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False, batched=batched)
    print('Baseline predictions', predictions[:10])
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    '''
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
    )
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
        accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
    
    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)
    '''
    return accuracies, {}


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    batched: bool = False,
    num_examples: int = 5
) -> None:
    seed_everything(41)
    print("Evaluating model:", model_type, model_variant, num_examples)

    results_file = get_results_file_path(model_type, model_variant, num_examples=num_examples,
                                         experiment_id=experiment_id, batched=batched)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    print('GPU available: {}'.format(torch.cuda.is_available()))
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant, load_to_cpu=False if torch.cuda.is_available() else True)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        try:

            if task_name in results:
                print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
                continue
            results[task_name] = {}

            print("\n" + "=" * 50)
            print(f"Running task {i+1}/{len(tasks)}: {task_name}")

            tic = time.time()
            accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples, batched)

            print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
            print(f"ICL Accuracy: {accuracies['icl']:.2f}")
            # print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
            # print(f"Dev Accuracy by layer: ", end="")
            # for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            #     print(f"{layer}: {accuracy:.2f}, ", end="")
            # print()
            print("Time:", time.time() - tic)

            results[task_name] = {
                "baseline_accuracy": accuracies["baseline"],
                "num_examples": num_examples,
                "icl_accuracy": accuracies["icl"],
                # "tv_accuracy": accuracies["tv"],
                # "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
                # "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
            }
        except ValueError as e:
            print(f'Encountered error in task {task_name}: {e}')

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
            batched = True
        elif len(sys.argv) >= 3:
            model_type, model_variant, batched, num_examples = sys.argv[1:]
            batched = eval(batched)
            num_examples = eval(num_examples)
        run_main_experiment(model_type, model_variant, num_examples=num_examples, batched=batched)


if __name__ == "__main__":
    main()
