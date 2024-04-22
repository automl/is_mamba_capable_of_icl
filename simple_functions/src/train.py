import math
import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
from torch import nn
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb
import random
import numpy as np

torch.backends.cudnn.benchmark = True


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    if args.training.learning_rate_schedule == "linear":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=100, gamma=0.9
        )
    elif args.training.learning_rate_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.training.train_steps)
        print(lr_scheduler)
        print(args.training.learning_rate)
    else:
        # constant
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
    device = args.device
    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims, **args.training.data_kwargs)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    ema_last_loss = float("inf")
    best_ema_last_loss = float("inf")
    ema_weight = 2/(1000 + 1) # roughly consider the last 1000 steps
    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)
    
        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        point_wise_loss_dict = dict(
            zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())
        )
        if i == 0:
            total_n_params = sum(p.numel() for p in model.parameters())


        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": point_wise_loss_dict,
                    "ema_last_loss": ema_last_loss,
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "total_n_params": total_n_params,
                },
                step=i,
            )

        curriculum.update()
        lr_scheduler.step()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

        if args.training['curriculum'].points.end-1 in point_wise_loss_dict:
            pointwise_end_point_loss = point_wise_loss_dict[args.training['curriculum'].points.end-1]
            if math.isinf(ema_last_loss):
                ema_last_loss = pointwise_end_point_loss
            ema_last_loss = ema_weight*ema_last_loss + (1- ema_weight)*pointwise_end_point_loss
            if i % args.training.save_every_steps == 0 and not args.test_run:
                if ema_last_loss < best_ema_last_loss:
                    best_ema_last_loss = ema_last_loss
                    torch.save(model.state_dict(), os.path.join(
                        args.out_dir, f"model_best_ema_last_loss.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            mode='online'
        )

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        # for mac
        device = 'mps'
    else:
        device = 'cpu'
    model = build_model(args.model)
    print("Number of model trainable parameters")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Number of model parameters")
    print(sum(p.numel() for p in model.parameters()))
    model.to(device)
    model.train()
    args['device'] = device

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "mamba", "s4", "gla"]
    print(f"Running with: {args}")
    seed_everything(args.training.seed)

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
