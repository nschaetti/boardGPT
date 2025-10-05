import argparse

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import MeanMetric
from transformers import AutoTokenizer
from boardGPT.nn import BoardMLPProbe, BoardLinearProbe
from boardGPT.models import GameGPT
from boardGPT.datasets import BoardDataset, collate_fn_board
from boardGPT.games.othello import game_to_board



def infinite_loader(
        dataloader: torch.utils.data.DataLoader
):
    """
    Infinite loader iterator.
    """
    while True:
        for batch in dataloader:
            X, Y = batch
            yield X, Y
        # end for
    # end while
# end def infinite_loader


def create_dataset(
        args,
        tokenizer: AutoTokenizer,
):
    train_dataset = BoardDataset(
        data_dir=args.data_dir,
        board_func=game_to_board,
        split="train"
    )

    val_dataset = BoardDataset(
        data_dir=args.data_dir,
        board_func=game_to_board,
        split="val"
    )

    # Create a dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_fn_board(b, tokenizer)
    )

    # Create a dataloader
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_fn_board(b, tokenizer)
    )

    return train_dataloader, val_dataloader
# end def create_dataset


def evaluate_model(
        args,
        model: nn.Module,
        probe: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
):
    val_iter = infinite_loader(dataloader)
    criterion = nn.CrossEntropyLoss()
    metric_acc = MulticlassAccuracy(num_classes=args.n_classes).to(device)
    metric_loss = MeanMetric().to(device)
    with torch.no_grad():
        for iter_i in range(args.n_val_iter):
            batch = next(val_iter)

            # Moves and game state
            move_idx, game_x = batch
            move_idx, game_x = move_idx.to(device), game_x.to(device)

            _, _, _, residuals = model(
                idx=move_idx,
                to_return=[
                    f'residuals{args.layer_i}'
                ]
            )

            # Stack residuals
            residuals = residuals[0]
            preds = probe(residuals)  # [B, 60, 64, 3]
            B, T, S, C = preds.shape  # 512, 60, 64, 3

            # -------------
            # 3. Flatten for loss
            # -------------
            preds = preds.view(B, T * S, C)  # [B, 30720, 3]
            targets = game_x.unsqueeze(1).expand(-1, T, -1)  # [B, 480, 64]
            targets = targets.reshape(B, -1)  # [B, 30720]

            # -------------
            # 4. Compute loss
            # -------------
            # preds: [B, 3840, 3]
            # targets: [B, 3840]
            loss = criterion(
                preds.reshape(B * T * S, 3),  # [B*30720, 3]
                targets.reshape(B * T * S).long()  # [B*30720]
            )

            # Update metrics
            metric_loss.update(loss)
            pred_labels = preds.argmax(dim=-1)  # [B, 30720]
            metric_acc.update(pred_labels.reshape(-1), targets.reshape(-1))
        # end for
    # end with

    # Show result
    epoch_loss = metric_loss.compute().item()
    epoch_acc = metric_acc.compute().item()
    return epoch_loss, epoch_acc
# end def evaluate_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--probe-type", type=str, required=True)
    parser.add_argument("--layer-i", type=int, required=True)
    parser.add_argument("--board-size", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--n-iter", type=int, required=True)
    parser.add_argument("--log-iter", type=int, required=True)
    parser.add_argument("--val-time", type=int, required=True)
    parser.add_argument("--n-val-iter", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--residual-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repo-id", type=str, default="theartificialis/OthelloGPT-Synthetic-20m")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="othello-gpt")
    parser.add_argument("--wandb-run-name", type=str, default="synthetic-othello-probe")
    args = parser.parse_args()
    return args
# end get_args


def main(args):
    """
    Main entry point.
    """
    # Get a device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, using CPU instead.")
    # end if

    # Get a GPT model
    model, model_config = GameGPT.from_pretrained(repo_id=args.repo_id)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, subfolder="tokenizer")
    model = model.to(device)
    model.eval()

    # Get dataloaders
    train_dataloader, val_dataloader = create_dataset(args, tokenizer)

    # Create model
    if args.probe_type.lower() == "mlp":
        probe = BoardMLPProbe(
            d_model=args.residual_size,
            board_size=args.board_size,
            n_classes=args.n_classes
        )
    elif args.probe_type.lower() == "linear":
        probe = BoardLinearProbe(
            d_model=args.residual_size,
            board_size=args.board_size,
            n_classes=args.n_classes
        )
    # end if
    probe.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    # Iterator
    train_iter = infinite_loader(train_dataloader)

    # Mode train
    probe.train()

    # Set up wandb logging if enabled
    if args.wandb:
        import wandb
        wandb_config = {
            "residual_size": args.residual_size,
            "board_size": args.board_size,
            "n_classes": args.n_classes,
            "n_iter": args.n_iter,
            "lr": args.lr,
            "device": args.device,
            "probe_type": args.probe_type,
            "repo_id": args.repo_id,
            "data_dir": args.data_dir,
        }
        assert args.wandb_project is not None, "WandB project must be provided."
        assert args.wandb_run_name is not None, "WandB run name must be provided."
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=wandb_config
        )
    # end if

    # Metrics on GPUs
    metric_acc = MulticlassAccuracy(num_classes=args.n_classes).to(device)
    metric_loss = MeanMetric().to(device)

    for iter_i in range(args.n_iter):
        batch = next(train_iter)

        # Moves and game state
        move_idx, game_x = batch

        # To cuda
        move_idx, game_x = move_idx.to(device), game_x.to(device)

        # Get residuals
        with torch.no_grad():
            _, _, _, residuals = model(
                idx=move_idx,
                to_return=[
                    f'residuals{args.layer_i}'
                ]
            )
        # end with

        # Stack residuals
        residuals = residuals[0]

        # -------------
        # 2. Forward probes
        # -------------
        # Inputs: [B, 60, 512]
        preds = probe(residuals)          # [B, 60, 64, 3]
        B, T, S, C = preds.shape        # 512, 60, 64, 3

        # -------------
        # 3. Flatten for loss
        # -------------
        preds = preds.view(B, T * S, C)           # [B, 30720, 3]
        targets = game_x.unsqueeze(1).expand(-1, T, -1)  # [B, 480, 64]
        targets = targets.reshape(B, -1)              # [B, 30720]

        # -------------
        # 4. Compute loss
        # -------------
        # preds: [B, 3840, 3]
        # targets: [B, 3840]
        loss = criterion(
            preds.reshape(B * T * S, 3),      # [B*30720, 3]
            targets.reshape(B * T * S).long()  # [B*30720]
        )

        # -------------
        # 5. Backprop
        # -------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        metric_loss.update(loss)
        pred_labels = preds.argmax(dim=-1)  # [B, 30720]
        metric_acc.update(pred_labels.reshape(-1), targets.reshape(-1))

        # Show current
        if iter_i % args.log_iter == 0:
            acc_val = metric_acc.compute().item()
            loss_val = metric_loss.compute().item()
            print(f"Iteration {iter_i}/1000 - loss: {loss_val:.4f}, acc: {acc_val:.4f}")

            # Log to wandb if enabled
            if args.wandb:
                log_data = {
                    "iter": iter_i,
                    "train/loss": loss_val,
                    "train/acc": acc_val,
                }
                wandb.log(log_data)
            # end if

            metric_acc.reset()
            metric_loss.reset()
        # end if

        # Eval
        if iter_i % args.val_time == 0:
            val_loss, val_acc = evaluate_model(
                args=args,
                model=model,
                probe=probe,
                dataloader=val_dataloader,
                device=device
            )
            print(f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            if args.wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                })
            # end if
        # end if
    # end for

    # Show result
    epoch_loss = metric_loss.compute().item()
    epoch_acc = metric_acc.compute().item()
    print(f"Epoch done - loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")
    if args.wandb:
        wandb.summary["loss"] = epoch_loss
        wandb.summary["acc"] = epoch_acc
    # end if
    metric_acc.reset()
    metric_loss.reset()
# end def main


if __name__ == "__main__":
    args = get_args()
    main(args)
# end if
