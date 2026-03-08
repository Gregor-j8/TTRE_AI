import argparse
from src.ml.v5.trainer import TrainerV5


def main():
    parser = argparse.ArgumentParser(description='V5 Mixed Training')
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy', type=float, default=0.05)
    parser.add_argument('--entropy-decay', type=float, default=0.9995)
    parser.add_argument('--min-entropy', type=float, default=0.01)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--v3-model', type=str, default='model_data/v3_good/model_v3_Final.pt')
    parser.add_argument('--v4-model', type=str, default='model_data/v4/model_v4_BEST_64pct_heuristic.pt')
    args = parser.parse_args()

    print("=" * 60)
    print("V5: MIXED TRAINING (TTRModelV5 Architecture)")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  LR: {args.lr}")
    print(f"  Entropy: {args.entropy} (decay={args.entropy_decay}, min={args.min_entropy})")
    print()
    print("  Opponent Mix:")
    print("    20% Self-play")
    print("    20% Heuristic")
    print("    15% OverallGame")
    print("    15% Blitz")
    print("    15% V3")
    print("    15% V4")
    print("=" * 60)

    trainer = TrainerV5(
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        entropy_decay=args.entropy_decay,
        min_entropy=args.min_entropy
    )

    try:
        trainer.load_v3_opponent(args.v3_model)
    except Exception as e:
        print(f"Warning: Could not load V3 opponent: {e}")

    try:
        trainer.load_v4_opponent(args.v4_model)
    except Exception as e:
        print(f"Warning: Could not load V4 opponent: {e}")

    if args.resume:
        trainer.load(args.resume)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        trainer.entropy_coef = args.entropy
        print(f"Resumed from: {args.resume}")

    trainer.train_mixed(
        num_episodes=args.episodes,
        eval_every=args.eval_every
    )


if __name__ == "__main__":
    main()
