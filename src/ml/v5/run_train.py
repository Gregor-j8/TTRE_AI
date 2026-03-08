import argparse
from src.ml.v5.trainer import TrainerV5


def main():
    parser = argparse.ArgumentParser(description='V5 Pure Self-Play Training')
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy', type=float, default=0.05)
    parser.add_argument('--entropy-decay', type=float, default=0.9995)
    parser.add_argument('--min-entropy', type=float, default=0.01)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    print("=" * 60)
    print("V5: PURE SELF-PLAY (TTRModelV3 Architecture)")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Entropy: {args.entropy} (decay={args.entropy_decay}, min={args.min_entropy})")
    print("=" * 60)
    print()
    print("Why Pure Self-Play?")
    print("  - V1 achieved 100% vs Heuristic with this approach")
    print("  - No imitation->RL transition to break the model")
    print("  - Policy + value learn together from the start")
    print("=" * 60)

    trainer = TrainerV5(
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        entropy_decay=args.entropy_decay,
        min_entropy=args.min_entropy
    )

    if args.resume:
        trainer.load(args.resume)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Resumed from: {args.resume}")

    trainer.train(
        num_episodes=args.episodes,
        eval_every=args.eval_every
    )


if __name__ == "__main__":
    main()
