import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='V3 Full Training Pipeline')
    parser.add_argument('--games', type=int, default=2000, help='Games for data collection')
    parser.add_argument('--imitation-epochs', type=int, default=15, help='Imitation epochs')
    parser.add_argument('--selfplay-episodes', type=int, default=20000, help='Self-play episodes')
    parser.add_argument('--skip-data', action='store_true', help='Skip data collection (use existing)')
    parser.add_argument('--skip-imitation', action='store_true', help='Skip imitation (use existing)')
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 70)
    print("V3 FULL TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Data collection: {args.games} games")
    print(f"  Imitation: {args.imitation_epochs} epochs")
    print(f"  Self-play: {args.selfplay_episodes} episodes")
    print("=" * 70)

    FIRST_ITER_PATH = 'model_data/v1_models/first_iteration/model_final.pt'
    DATA_PATH = 'first_iter_data_v3.pt'

    if not args.skip_data:
        print("\n" + "=" * 70)
        print("STEP 1: DATA COLLECTION (from First Iteration)")
        print("=" * 70)

        from src.ml.v3.data_collector import FirstIterationDataCollector

        collector = FirstIterationDataCollector(FIRST_ITER_PATH)
        start = time.time()
        all_data = collector.collect_games(args.games)
        elapsed = time.time() - start

        print(f"\nCollection completed in {elapsed:.1f}s")
        print(f"Total examples: {len(all_data)}")
        print(f"Unique actions: {len(collector.action_to_idx)}")

        collector.save_data(all_data, DATA_PATH)
    else:
        print(f"\n[Skipping data collection - using existing {DATA_PATH}]")

    if not args.skip_imitation:
        print("\n" + "=" * 70)
        print("STEP 2: IMITATION LEARNING (from First Iteration)")
        print("=" * 70)

        from src.ml.v3.imitation_trainer import ImitationTrainerV3

        trainer = ImitationTrainerV3(lr=1e-3)
        trainer.load_data(DATA_PATH)
        trainer.load_first_iteration(FIRST_ITER_PATH)

        print(f"\nModel parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

        print("\n--- Initial Evaluation ---")
        init_heuristic = trainer.evaluate_vs_heuristic(100)
        init_first_iter = trainer.evaluate_vs_first_iteration(100)
        print(f"vs Heuristic: {init_heuristic*100:.1f}%")
        print(f"vs First Iteration: {init_first_iter*100:.1f}%")

        start = time.time()
        best_first_iter = 0

        for epoch in range(args.imitation_epochs):
            loss, accuracy = trainer.train_epoch(batch_size=64)
            vs_heuristic = trainer.evaluate_vs_heuristic(50)
            vs_first_iter = trainer.evaluate_vs_first_iteration(50)
            print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Acc={accuracy*100:.1f}%, vs Heuristic={vs_heuristic*100:.1f}%, vs First Iter={vs_first_iter*100:.1f}%")

            if vs_first_iter > best_first_iter:
                best_first_iter = vs_first_iter
                trainer.save('imitation_model_v3.pt')

        elapsed = time.time() - start
        print(f"\nImitation completed in {elapsed/60:.1f} minutes")
        print(f"Best vs First Iteration: {best_first_iter*100:.1f}%")
    else:
        print("\n[Skipping imitation - using existing imitation_model_v3.pt]")

    print("\n" + "=" * 70)
    print("STEP 3: MIXED TRAINING (Self-Play + vs First Iteration)")
    print("=" * 70)

    from src.ml.v3.trainer import TrainerV3

    trainer = TrainerV3(
        lr=1e-4,
        batch_size=128,
        gamma=0.99,
        entropy_coef=0.05
    )

    FIRST_ITER_PATH = 'model_data/v1_models/first_iteration/model_final.pt'
    trainer.load_opponent(FIRST_ITER_PATH)

    trainer.load('imitation_model_v3.pt')
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 1e-4
    print(f"Loaded imitation checkpoint (reset lr to 1e-4)")

    trainer.train(
        num_episodes=args.selfplay_episodes,
        eval_every=2000
    )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Final model saved to: model_v3_final.pt")


if __name__ == "__main__":
    main()
