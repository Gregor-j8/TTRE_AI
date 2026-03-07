import time
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='V4 Full Training Pipeline')
    parser.add_argument('--overall-games', type=int, default=2000)
    parser.add_argument('--blitz-games', type=int, default=1000)
    parser.add_argument('--v3-games', type=int, default=500)
    parser.add_argument('--ticket-weight', type=float, default=2.0)
    parser.add_argument('--imitation-epochs', type=int, default=15)
    parser.add_argument('--rl-episodes', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--imitation-lr', type=float, default=5e-4)
    parser.add_argument('--rl-lr', type=float, default=1e-4)
    parser.add_argument('--entropy-coef', type=float, default=0.1)
    parser.add_argument('--entropy-decay', type=float, default=0.995)
    parser.add_argument('--skip-data', action='store_true')
    parser.add_argument('--skip-imitation', action='store_true')
    parser.add_argument('--data-path', type=str, default='v4_training_data.pt')
    parser.add_argument('--imitation-output', type=str, default='model_v4_imitation.pt')
    parser.add_argument('--final-output', type=str, default='model_v4_final.pt')
    args = parser.parse_args()

    total_start = time.time()
    total_games = args.overall_games + args.blitz_games + args.v3_games

    V3_MODEL_PATH = 'model_data/v3_good/model_v3_Final.pt'

    print("=" * 70)
    print("V4 FULL TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Model: 10M params (hidden=704, layers=6)")
    print()
    print("  PHASE 1 - Data Collection:")
    print(f"    OverallGame: {args.overall_games} games")
    print(f"    Blitz: {args.blitz_games} games")
    print(f"    V3: {args.v3_games} games")
    print(f"    Ticket weight: {args.ticket_weight}x")
    print()
    print("  PHASE 2 - Imitation Learning:")
    print(f"    Epochs: {args.imitation_epochs}")
    print(f"    LR: {args.imitation_lr}")
    print()
    print("  PHASE 3 - Reinforcement Learning:")
    print(f"    Episodes: {args.rl_episodes}")
    print(f"    LR: {args.rl_lr}")
    print(f"    Entropy: {args.entropy_coef} (decay={args.entropy_decay})")
    print("=" * 70)

    if not args.skip_data:
        print("\n" + "=" * 70)
        print("PHASE 1: DATA COLLECTION")
        print("=" * 70)

        from src.ml.v4.data_collector import MultiTeacherDataCollector

        collector = MultiTeacherDataCollector(V3_MODEL_PATH)

        start = time.time()
        all_data, stats = collector.collect_all(
            overall_game_games=args.overall_games,
            blitz_games=args.blitz_games,
            v3_games=args.v3_games,
            ticket_weight=args.ticket_weight
        )
        elapsed = time.time() - start

        print(f"\nCollection completed in {elapsed:.1f}s")
        print(f"Total examples: {len(all_data)}")

        collector.save_data(all_data, args.data_path)
    else:
        print(f"\n[Skipping data collection - using {args.data_path}]")

    if not args.skip_imitation:
        print("\n" + "=" * 70)
        print("PHASE 2: IMITATION LEARNING")
        print("=" * 70)

        from src.ml.v4.imitation_trainer import ImitationTrainerV4

        trainer = ImitationTrainerV4(lr=args.imitation_lr, label_smoothing=0.1)
        trainer.load_data(args.data_path)

        print("\n--- Initial Evaluation ---")
        init_random = trainer.evaluate_vs_random(100)
        init_heuristic = trainer.evaluate_vs_heuristic(100)
        init_overall = trainer.evaluate_vs_overall_game(50)
        init_blitz = trainer.evaluate_vs_blitz(50)
        print(f"vs Random: {init_random*100:.1f}%")
        print(f"vs Heuristic: {init_heuristic*100:.1f}%")
        print(f"vs OverallGame: {init_overall*100:.1f}%")
        print(f"vs Blitz: {init_blitz*100:.1f}%")

        print("\n--- Training ---")
        start = time.time()
        best_combined = 0

        for epoch in range(args.imitation_epochs):
            loss, accuracy = trainer.train_epoch(
                batch_size=args.batch_size,
                use_weights=True
            )

            vs_random = trainer.evaluate_vs_random(50)
            vs_heuristic = trainer.evaluate_vs_heuristic(50)
            vs_overall = trainer.evaluate_vs_overall_game(50)
            vs_blitz = trainer.evaluate_vs_blitz(50)

            combined = vs_overall + vs_blitz + vs_heuristic

            print(f"Ep {epoch+1:2d}: Loss={loss:.4f} Acc={accuracy*100:.1f}% | "
                  f"Rand={vs_random*100:.0f}% Heur={vs_heuristic*100:.0f}% "
                  f"Over={vs_overall*100:.0f}% Blitz={vs_blitz*100:.0f}%")

            if combined > best_combined:
                best_combined = combined
                trainer.save(args.imitation_output)

        elapsed = time.time() - start
        print(f"\nImitation completed in {elapsed/60:.1f} minutes")
    else:
        print(f"\n[Skipping imitation - using {args.imitation_output}]")

    print("\n" + "=" * 70)
    print("PHASE 3: REINFORCEMENT LEARNING")
    print("=" * 70)

    from src.ml.v4.trainer import TrainerV4

    rl_trainer = TrainerV4(
        lr=args.rl_lr,
        batch_size=args.batch_size,
        gamma=0.99,
        entropy_coef=args.entropy_coef,
        entropy_decay=args.entropy_decay,
        min_entropy=0.01
    )

    rl_trainer.load_v3_opponent(V3_MODEL_PATH)

    rl_trainer.load(args.imitation_output)
    for param_group in rl_trainer.optimizer.param_groups:
        param_group['lr'] = args.rl_lr
    print(f"Loaded imitation checkpoint (reset lr to {args.rl_lr})")

    rl_trainer.train(
        num_episodes=args.rl_episodes,
        eval_every=2000
    )

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print()
    print("Output files:")
    print(f"  Data:      {args.data_path}")
    print(f"  Imitation: {args.imitation_output}")
    print(f"  Final:     model_v4_final.pt")


if __name__ == "__main__":
    main()
