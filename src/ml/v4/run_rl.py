import argparse
from src.ml.v4.trainer import TrainerV4


def main():
    parser = argparse.ArgumentParser(description='V4 RL Training')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--entropy', type=float, default=0.01)
    parser.add_argument('--entropy-decay', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--imitation-model', type=str, default='model_v4_imitation.pt')
    parser.add_argument('--v3-model', type=str, default='model_data/v3_good/model_v3_Final.pt')
    parser.add_argument('--eval-every', type=int, default=2000)
    args = parser.parse_args()

    print("=" * 60)
    print("V4 RL TRAINING (Low Entropy)")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  Entropy: {args.entropy} (decay={args.entropy_decay})")
    print(f"  LR: {args.lr}")
    print(f"  Imitation model: {args.imitation_model}")
    print("=" * 60)

    trainer = TrainerV4(
        lr=args.lr,
        batch_size=128,
        gamma=0.99,
        entropy_coef=args.entropy,
        entropy_decay=args.entropy_decay,
        min_entropy=0.001
    )

    trainer.load_v3_opponent(args.v3_model)
    trainer.load(args.imitation_model)

    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.lr

    print(f"Loaded {args.imitation_model}, starting RL\n")

    trainer.train(num_episodes=args.episodes, eval_every=args.eval_every)


if __name__ == "__main__":
    main()
