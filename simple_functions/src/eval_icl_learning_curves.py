import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
import torch.cuda
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from eval import get_model_from_run
from nethook import TraceDict
from samplers import get_data_sampler
from tasks import get_task_sampler


# Import function to split into train and test split


def eval_icl_learning_curve(model_type: str, probing: bool = False, data: str = 'gaussian', task_name: str = 'linear_regression'):
    if model_type == 'transformer':
        # transformer
        if task_name == 'linear_regression':
            model, conf = get_model_from_run('path')
        elif task_name == 'relu_2nn_regression':
            model, conf = get_model_from_run('path')
        elif task_name == 'decision_tree':
            model, conf = get_model_from_run('path')
        decoder = lambda x: model._read_out(model._backbone.ln_f(x))
        layers = [f'_backbone.h.{i}' for i in range(12)]

    else:
        # mamba
        if task_name == 'linear_regression':
            model, conf = get_model_from_run('path')
        elif task_name == 'relu_2nn_regression':
            model, conf = get_model_from_run('path')
        elif task_name == 'decision_tree':
            model, conf = get_model_from_run('path')
        from mamba_ssm.ops.triton.layernorm import rms_norm_fn
        fused_add_norm_fn = lambda hidden_state, residual: (
            rms_norm_fn(hidden_state, model._backbone.backbone.norm_f.weight, model._backbone.backbone.norm_f.bias,
                        eps=model._backbone.backbone.norm_f.eps, residual=residual, prenorm=False,
                        residual_in_fp32=model._backbone.backbone.residual_in_fp32))
        lm_head = model._backbone.lm_head
        decoder = lambda hidden_state, residual: lm_head(fused_add_norm_fn(hidden_state, residual))
        layers = [f'_backbone.backbone.layers.{i}' for i in range(24)]

    model.eval()

    batch_size = 64
    n_dims = 20
    n_points = 70
    n_train_points = 40
    n_points_lr = 10
    test_size = (n_points - n_train_points) / n_points

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_sampler = get_data_sampler(data, n_dims=n_dims, **{'exp': 1} if data == 'skewed_gaussian' else {})
    task = get_task_sampler(task_name, n_dims, batch_size)(**{})

    model.to(device)

    xs = data_sampler.sample_xs(n_points, b_size=batch_size)
    ys = task.evaluate(xs)

    preds = []
    targets = []

    decoded_hidden_representation = defaultdict(list)

    for i in tqdm(range(n_train_points, n_points), desc='Evaluating test points based on training context.'):
        train_xs, train_ys = xs[:, :n_train_points, :], ys[:, :n_train_points]
        test_xs, test_ys = xs[:, i, :], ys[:, i]
        xs_comb, ys_comb = torch.concatenate([train_xs, test_xs.unsqueeze(dim=1)], dim=1), torch.concatenate(
            [train_ys, test_ys.unsqueeze(dim=1)], dim=1)
        with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:
            with torch.no_grad():
                pred = model(xs_comb.to(device), ys_comb.to(device)).detach()
        preds.append(pred[:, -1].unsqueeze(-1).detach())
        targets.append(test_ys.unsqueeze(-1).detach())
        for k, v in td.items():
            if model_type == 'mamba':
                decoded_hidden_representation[k].append(decoder(v.output[0], v.output[1]))
            elif model_type == 'transformer':
                decoded_hidden_representation[k].append(decoder(v.output[0]))
            if k == layers[-1]:
                assert torch.allclose(decoded_hidden_representation[k][-1][:, -2].flatten(),
                                      pred[:, -1].flatten()), 'Decoding was unsuccessful'

    decoded_predictions = {}
    for layer, decoded_h in decoded_hidden_representation.items():
        train_y_pred = None
        test_y_pred = []
        for i, d_h_i in enumerate(decoded_h):
            if i == 0:
                train_y_pred = d_h_i[:, :-2:2].squeeze(-1)
            test_y_pred.append(d_h_i[:, -2].squeeze(-1))
        decoded_predictions[layer] = torch.concatenate([train_y_pred, torch.stack(test_y_pred, dim=-1)], dim=-1)

    per_task_errors = torch.nn.MSELoss(reduction='none')(
        torch.concatenate(preds, dim=-1).to(device),
        torch.concatenate(targets, dim=-1).to(device)).mean(axis=-1)
    layer_error = []
    ys = ys.detach().cpu().numpy()

    # Get the activations from the all the layers in the transformer, which are structured '_backbone.h.{i}'
    for layer, v in decoded_predictions.items():
        # Split into train and test set out and ys
        for i in range(batch_size):
            out_down_proj = v[i]

            out_train, out_test = out_down_proj[n_train_points - n_points_lr:n_train_points], out_down_proj[
                                                                                              n_train_points:]
            ys_train, y_test = ys[i][n_train_points - n_points_lr:n_train_points], ys[i][n_train_points:]
            # Flatten the activations
            lr_model = LinearRegression()
            lr_model.fit(out_train.reshape(-1, out_train.shape[-1]).detach().cpu().numpy().T, ys_train.reshape(-1, 1))
            # Predict on the test set
            y_pred = lr_model.predict(out_test.reshape(-1, 1).detach().cpu().numpy())

            # Calculate the MSE
            if probing:
                mse = mean_squared_error(y_pred, y_test.reshape(-1, 1))
            else:
                mse = mean_squared_error(out_down_proj.reshape(-1, 1)[n_train_points:].flatten().detach().cpu().numpy(),
                                         ys[i][n_train_points:].flatten())
            # print(f'Layer {layer} MSE: {mse}')
            layer_error.append([layer, mse, lr_model.coef_.tolist(), lr_model.intercept_.tolist()])
        print(f'Layer {layer} - Activation Shape: {v.shape}')

    # Save the dataframe with the MSE for each layer
    df = pd.DataFrame(layer_error, columns=['layer', 'mse', 'coef', 'intercept'])
    os.makedirs(f'../results/{task_name}/{model_type}', exist_ok=True)
    df.to_csv(
        f'../results/{task_name}/{model_type}/decoder_probing_{probing}_n_points_lr_{n_points_lr}_mse_{data}.csv')


def gradient_descent_linear_regression(data: str = 'gaussian'):
    batch_size = 64
    n_dims = 20
    n_points = 70
    n_train_points = 40
    data_sampler = get_data_sampler(data, n_dims=n_dims, **{'exp': 1} if data == 'skewed_gaussian' else {})
    task = get_task_sampler('linear_regression', n_dims, batch_size)(**{})

    xs = data_sampler.sample_xs(n_points, b_size=batch_size)
    ys = task.evaluate(xs)

    # Split into train and test
    train_xs, train_ys = xs[:, :n_train_points, :], ys[:, :n_train_points]
    test_xs, test_ys = xs[:, n_train_points:, :], ys[:, n_train_points:]

    def square_loss(x, y, w):
        return 0.5 * torch.mean((x @ w - y) ** 2)

    def square_loss_gradient(x, y, w):
        return (1 / len(y)) * x.T @ (x @ w - y)

    def compute_Lmu(x):
        eigenvalues, _ = np.linalg.eigh(x.T @ x / x.shape[0])
        L = np.max(eigenvalues)
        mu = np.maximum(0, np.min(eigenvalues))
        return L, mu

    def compute_opt_ridge_step_size(alpha_l2, L=None, mu=None):
        # alpha_l2 is the l2-regularization parameter
        return 2 / (L + mu + 2 * alpha_l2)

    def gradient_descent(train_xs, train_ys,
                         test_xs, test_ys,
                         preconditioned: bool = False, n_iter: int = 500, gamma: float = 0.0001939212572065831) \
            -> pd.DataFrame:
        per_task_losses = {}
        for task in range(batch_size):
            per_task_losses[task] = []
            H = torch.eye(n_dims) - gamma * train_xs[task].T @ train_xs[task] if preconditioned else torch.eye(n_dims)
            w = torch.zeros(size=(n_dims,))
            for iter in range(n_iter):
                # compute the loss and convert to numpy
                loss = square_loss(test_xs[task] @ H, test_ys[task], w).detach().cpu().numpy()
                per_task_losses[task].append(float(loss) / n_dims * 2)
                w -= square_loss_gradient(train_xs[task] @ H, train_ys[task], w) * compute_opt_ridge_step_size(0.0, *compute_Lmu(train_xs[task]))
        return pd.DataFrame(per_task_losses)

    df_gd = gradient_descent(train_xs, train_ys, test_xs, test_ys, preconditioned=False)
    df_prec = gradient_descent(train_xs, train_ys, test_xs, test_ys, preconditioned=True)

    '''
    losses, gammas = [], []
    for gamma in 10 ** np.linspace(-6, -3, 300):
        df_prec = gradient_descent(train_xs, train_ys, test_xs, test_ys, n_iter=22, preconditioned=True, gamma=gamma)
        losses.append(df_prec.iloc[-1, :].mean())
        gammas.append(gamma)

    print(gammas[np.argmin(losses)])
    plt.plot(gammas, losses)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('gamma')
    plt.ylabel('loss')
    plt.savefig('../results/hpo_prec_gd.pdf', bbox_inches='tight')
    '''
    # format the losses in a way seaborn likes
    df_gd = df_gd.melt(var_name='task', value_name='loss')
    # Add the number of iterations as a column
    df_gd['iteration'] = df_gd.groupby('task').cumcount()
    df_gd.to_csv('../results/gradient_descent_learning_curve.csv')

    # format the losses in a way seaborn likes
    df_prec = df_prec.melt(var_name='task', value_name='loss')
    # Add the number of iterations as a column
    df_prec['iteration'] = df_prec.groupby('task').cumcount()
    df_prec.to_csv('../results/prec_gradient_descent_learning_curve.csv')
    # df.to_csv('../results/gradient_descent_learning_curve.csv')

    # draw a nice plot of the losses
    plt.style.use(['science', 'no-latex', 'light'])
    plt.figure(figsize=(2.8, 2.2))
    plt.title('Gradient Descent Learning Curve')
    sns.lineplot(x='iteration', y='loss', data=df_gd, label='GD')
    sns.lineplot(x='iteration', y='loss', data=df_prec, label='Preconditioned GD')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig('../results/gradient_descent_learning_curve.pdf', bbox_inches='tight')


def plot_icl_learning_curve(task: str, probing: bool = False, n_points_lr: int = 10):
    if task == 'linear_regression':
        mamba_path = f'../results/linear_regression/mamba/decoder_probing_True_n_points_lr_10_mse_skewed_gaussian.csv'
        transformer_path = f'../results/linear_regression/transformer/decoder_probing_True_n_points_lr_10_mse_skewed_gaussian.csv'
    elif task == 'relu_2nn_regression':
        mamba_path = '../results/relu_2nn_regression/mamba/decoder_probing_True_n_points_lr_10_mse_gaussian.csv'
        transformer_path = '../results/relu_2nn_regression/transformer/decoder_probing_True_n_points_lr_10_mse_gaussian.csv'
    elif task == 'decision_tree':
        mamba_path = '../results/decision_tree/mamba/decoder_probing_True_n_points_lr_10_mse_gaussian.csv'
        transformer_path = '../results/decision_tree/transformer/decoder_probing_True_n_points_lr_10_mse_gaussian.csv'
    else:
        raise NotImplementedError(f'Unknown task {task}')

    df_mamba = pd.read_csv(mamba_path)
    df_mamba['layer'] = df_mamba['layer'].apply(lambda x: int(x.split('.')[-1]))
    df_mamba = df_mamba.sort_values(by='layer')
    df_mamba['layer'] = df_mamba['layer'] / df_mamba['layer'].max()
    df_mamba['Model'] = ['Mamba' for _ in range(len(df_mamba))]
    df_mamba['coef'] = df_mamba['coef'].apply(lambda x: eval(x)[0][0])
    df_mamba['intercept'] = df_mamba['intercept'].apply(lambda x: eval(x)[0])

    df_transformer = pd.read_csv(transformer_path)
    df_transformer['layer'] = df_transformer['layer'].apply(lambda x: int(x.split('.')[-1]))
    df_transformer = df_transformer.sort_values(by='layer')
    df_transformer['layer'] = df_transformer['layer'] / df_transformer['layer'].max()
    df_transformer['Model'] = ['Transformer' for _ in range(len(df_transformer))]
    df_transformer['coef'] = df_transformer['coef'].apply(lambda x: eval(x)[0][0])
    df_transformer['intercept'] = df_transformer['intercept'].apply(lambda x: eval(x)[0])

    df = pd.concat([df_mamba, df_transformer])
    plt.style.use(['science', 'no-latex', 'light'])
    fig, axs = plt.subplots(2, 1, figsize=(2.2, 3.5), sharex=True)

    # plt.title('ICL Learning Curve')
    sns.lineplot(x='layer', y='coef', data=df, hue='Model', ax=axs[0])
    sns.lineplot(x='layer', y='intercept', data=df, hue='Model', ax=axs[1])
    plt.xlabel('Ratio of layers')
    axs[0].set_ylabel('Scale')
    axs[1].set_ylabel('Shift')
    axs[0].grid(True, which="both", ls="-", alpha=0.3)
    axs[1].grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    # remove legend from axs[0]
    axs[0].legend().remove()
    # remove title from legend in axs[1]
    axs[1].legend(title='')
    plt.savefig(
        f'../plots/mamba_vs_transformer_decoder_probing_scale_shift_{task}_{probing}_n_points_lr_{n_points_lr}_mse.pdf',
        dpi=300, bbox_inches='tight')

    if task == 'linear_regression':
        df_gd = pd.read_csv('../results/gradient_descent_learning_curve.csv')
        df_gd = df_gd[df_gd['iteration'] < 24]
        df_gd['iteration'] /= max(df_gd['iteration'])
        # rename the columns
        df_gd.rename({'iteration': 'layer', 'loss': 'mse'}, axis=1, inplace=True)

        df_prec = pd.read_csv('../results/prec_gradient_descent_learning_curve.csv')
        df_prec = df_prec[df_prec['iteration'] < 24]
        df_prec['iteration'] /= max(df_prec['iteration'])
        df_prec.rename({'iteration': 'layer', 'loss': 'mse'}, axis=1, inplace=True)

        # Merge df_gd and df_prec
        df_gd['Model'] = ['GD' for _ in range(len(df_gd))]
        df_prec['Model'] = ['GD++' for _ in range(len(df_prec))]
        df_gd_prec = pd.concat([df_gd, df_prec])
        df = pd.concat([df, df_gd_prec])

    plt.style.use(['science', 'no-latex', 'light'])
    # Plot the MSE for each layer and make it a fancy plot
    plt.figure(figsize=(2.8, 2.2))
    # plt.title('ICL Learning Curve')
    sns.lineplot(x='layer', y='mse', data=df, hue='Model', style='Model', markers=True, n_boot=10000)
    plt.xlabel('Ratio of layers')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    # remove the legend title
    plt.legend(title='', loc='lower left')
    plt.yscale('log')
    plt.tight_layout()

    os.makedirs('../plots', exist_ok=True)
    plt.savefig(
        f'../plots/mamba_vs_transformer_decoder_probing_{task}_{probing}_n_points_lr_{n_points_lr}_mse.pdf',
        dpi=300, bbox_inches='tight')
    plt.close()

    # Make a plot of the coefficient and scale of the linear regression model comparing mamba and transformer
    # show them in one plot


def test_mamba_output():
    '''
    # transformer
    model, conf = get_model_from_run('../models_rerun_25_01_24/linear_regression/transformer_lr_25_01_24/')
    decoder = lambda x: model._read_out(model._backbone.ln_f(x))
    layers = [f'_backbone.h.11']
    '''
    # mamba
    model, conf = get_model_from_run('../models_rerun_25_01_24/linear_regression/mamba_lr_25_01_24/')
    layers = ['_backbone.backbone.layers.23']
    from mamba_ssm.ops.triton.layernorm import rms_norm_fn
    fused_add_norm_fn = lambda hidden_state, residual: rms_norm_fn(
        hidden_state, model._backbone.backbone.norm_f.weight, model._backbone.backbone.norm_f.bias,
        eps=model._backbone.backbone.norm_f.eps, residual=residual, prenorm=False,
        residual_in_fp32=model._backbone.backbone.residual_in_fp32)
    lm_head = model._backbone.lm_head
    decoder = lambda hidden_state, residual: lm_head(fused_add_norm_fn(hidden_state, residual))
    model.eval()

    batch_size = 1
    n_dims = 20
    n_points = 11
    n_train_points = 10

    device = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"

    data_sampler = get_data_sampler('gaussian', n_dims=n_dims)
    task = get_task_sampler('linear_regression', n_dims, batch_size)(**{})

    model.to(device)

    xs = data_sampler.sample_xs(n_points, b_size=batch_size)
    ys = task.evaluate(xs)

    i = n_train_points
    train_xs, train_ys = xs[:, :n_train_points, :], ys[:, :n_train_points]
    test_xs, test_ys = xs[:, i, :], ys[:, i]
    xs_comb, ys_comb = torch.concatenate([train_xs, test_xs.unsqueeze(dim=1)], dim=1), torch.concatenate(
        [train_ys, test_ys.unsqueeze(dim=1)], dim=1)
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:
        with torch.no_grad():
            pred = model(xs_comb.to(device), ys_comb.to(device)).detach()

    print(td[layers[0]].output[0].shape)
    print(td[layers[0]].output[1].shape)
    print('Decoding was successful:', torch.allclose(
        decoder(td[layers[0]].output[0], torch.zeros_like(td[layers[0]].output[1]))[0, -2, 0],
        pred[0, -1]))
    print('decoded prediction', decoder(td[layers[0]].output[0], td[layers[0]].output[1])[0, -2, 0])
    print('real prediction', pred[0, -1])
    print('difference ', (decoder(td[layers[0]].output[0], td[layers[0]].output[1])[0, -2, 0] - pred[0, -1]) * 10000)


if __name__ == "__main__":
    for model_type in ['transformer', 'mamba']:
        for probing in [True]:
            for data in ['gaussian']:
                for task in ['relu_2nn_regression', 'decision_tree']:
                    eval_icl_learning_curve(
                        model_type, probing,
                        'gaussian' if task in ['relu_2nn_regression', 'decision_tree', 'sparse_linear_regression']
                        else 'skewed_gaussian',
                        task
                    )

    for task in ['linear_regression', 'sparse_linear_regression', 'relu_2nn_regression', 'decision_tree']:
       for n_points_lr in [10]:
            plot_icl_learning_curve(task, probing=True, n_points_lr=n_points_lr)

