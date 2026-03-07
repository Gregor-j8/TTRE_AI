import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import random
import time
import argparse

from src.game import Game
from src.ml.v2.state_encoder import StateEncoderV2
from src.ml.v4.model import TTRModelV4
from src.ml.v4.trainer import TrainerV4, V3Opponent
from src.players import random_choose, ticket_focused_choose, overall_game_choose, blitz_choose


class StableTrainerV4(TrainerV4):
    def pretrain_value_head(self, num_games=500, epochs=5):
        print(f"\n--- Pretraining Value Head ({num_games} games, {epochs} epochs) ---")

        samples = []
        wins = 0

        self.model.eval()

        for game_num in range(num_games):
            game = Game(2)
            model_player = game_num % 2
            game_states = []

            turn = 0
            while not game.game_over and turn < 500:
                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx
                player = game.get_current_player()

                if player_idx == model_player:
                    data = self.encoder.encode_state(game.state, player_idx)
                    game_states.append(data)
                    action, _, _ = self.model_choose(
                        game.state, player, actions, game.board, player_idx, explore=False
                    )
                else:
                    action = ticket_focused_choose(game.state, player, actions, game.board)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                outcome = 1.0
                wins += 1
            elif scores[model_player] < scores[1 - model_player]:
                outcome = -1.0
            else:
                outcome = 0.0

            for data in game_states:
                samples.append({'data': data, 'value': outcome})

            if (game_num + 1) % 100 == 0:
                print(f"  Collected {game_num + 1}/{num_games} games, "
                      f"win rate: {wins/(game_num+1)*100:.1f}%")

        print(f"  Total samples: {len(samples)}, win rate: {wins/num_games*100:.1f}%")

        self.model.train()
        value_optimizer = optim.Adam(self.model.value_head.parameters(), lr=1e-3)

        for epoch in range(epochs):
            random.shuffle(samples)
            total_loss = 0
            num_batches = 0

            for i in range(0, len(samples), 64):
                batch_samples = samples[i:i+64]
                if len(batch_samples) < 8:
                    continue

                data_list = [s['data'] for s in batch_samples]
                targets = torch.tensor(
                    [s['value'] for s in batch_samples],
                    dtype=torch.float32,
                    device=self.device
                )

                batch = Batch.from_data_list(data_list).to(self.device)

                _, values = self.model(batch)
                values = values.squeeze(-1)

                loss = F.mse_loss(values, targets)

                value_optimizer.zero_grad()
                loss.backward()
                value_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Value pretrain epoch {epoch+1}: loss={avg_loss:.4f}")

        print("  Value head pretraining complete\n")

    def train_stable(self, num_episodes=10000, eval_every=2000, value_recalibrate_every=2000):
        print(f"Starting Stable RL training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Settings: lr={self.optimizer.param_groups[0]['lr']}, entropy=0, recalibrate every {value_recalibrate_every}")

        self.entropy_coef = 0

        print("\n--- Initial Evaluation ---")
        init_random = self.evaluate_vs_random(100)
        init_heuristic = self.evaluate_vs_heuristic(100)
        init_overall = self.evaluate_vs_overall_game(50)
        init_blitz = self.evaluate_vs_blitz(50)
        init_v3 = self.evaluate_vs_v3(50) if self.v3_opponent else None
        print(f"vs Random: {init_random*100:.1f}%")
        print(f"vs Heuristic: {init_heuristic*100:.1f}%")
        print(f"vs OverallGame: {init_overall*100:.1f}%")
        print(f"vs Blitz: {init_blitz*100:.1f}%")
        if init_v3 is not None:
            print(f"vs V3: {init_v3*100:.1f}%")

        print("\n" + "=" * 60)
        print("STABLE RL TRAINING (No entropy, low LR, value recalibration)")
        print("=" * 60)

        losses = []
        total_episodes = 0
        start_time = time.time()
        best_heuristic = init_heuristic

        while total_episodes < num_episodes:
            batch_start = time.time()
            all_trajectories, all_rewards = self.collect_episodes(self.batch_size)
            collect_time = time.time() - batch_start

            train_start = time.time()
            stats = self.train_on_batch(all_trajectories, all_rewards)
            train_time = time.time() - train_start

            losses.append(stats['loss'])
            total_episodes += len(all_trajectories)

            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            eps_per_sec = self.batch_size / (collect_time + train_time)
            print(f"Ep {total_episodes}: loss={avg_loss:.4f}, value_loss={stats['value_loss']:.4f}, "
                  f"{eps_per_sec:.1f} eps/sec", flush=True)

            if total_episodes % eval_every < self.batch_size:
                vs_random = self.evaluate_vs_random(100)
                vs_heuristic = self.evaluate_vs_heuristic(100)
                vs_overall = self.evaluate_vs_overall_game(50)
                vs_blitz = self.evaluate_vs_blitz(50)
                vs_v3 = self.evaluate_vs_v3(50) if self.v3_opponent else None
                elapsed = time.time() - start_time

                v3_str = f", V3={vs_v3*100:.0f}%" if vs_v3 is not None else ""
                print(f"  -> Rand={vs_random*100:.0f}%, Heur={vs_heuristic*100:.0f}%, "
                      f"Over={vs_overall*100:.0f}%, Blitz={vs_blitz*100:.0f}%{v3_str} ({elapsed/60:.1f}min)")

                if vs_heuristic > best_heuristic:
                    best_heuristic = vs_heuristic
                    self.save("model_v4_stable_best.pt")
                    print(f"  ** New best: {vs_heuristic*100:.1f}% vs Heuristic **")

                self.save(f"model_v4_stable_ep{total_episodes}.pt")

            if total_episodes % value_recalibrate_every < self.batch_size and total_episodes > 0:
                print("\n  Recalibrating value head...")
                self.pretrain_value_head(num_games=200, epochs=2)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        final_random = self.evaluate_vs_random(200)
        final_heuristic = self.evaluate_vs_heuristic(200)
        final_overall = self.evaluate_vs_overall_game(100)
        final_blitz = self.evaluate_vs_blitz(100)
        final_v3 = self.evaluate_vs_v3(100) if self.v3_opponent else None

        print(f"vs Random:      {final_random*100:.1f}%")
        print(f"vs Heuristic:   {final_heuristic*100:.1f}%")
        print(f"vs OverallGame: {final_overall*100:.1f}%")
        print(f"vs Blitz:       {final_blitz*100:.1f}%")
        if final_v3 is not None:
            print(f"vs V3:          {final_v3*100:.1f}%")

        self.save("model_v4_stable_final.pt")

        print(f"\nBest vs Heuristic during training: {best_heuristic*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='V4 Stable RL Training')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5, help='Very low LR for stability')
    parser.add_argument('--value-pretrain-games', type=int, default=500)
    parser.add_argument('--value-recalibrate-every', type=int, default=2000)
    parser.add_argument('--imitation-model', type=str, default='model_v4_imitation.pt')
    parser.add_argument('--v3-model', type=str, default='model_data/v3_good/model_v3_Final.pt')
    parser.add_argument('--eval-every', type=int, default=2000)
    args = parser.parse_args()

    print("=" * 60)
    print("V4 STABLE RL TRAINING")
    print("=" * 60)
    print(f"  Episodes: {args.episodes}")
    print(f"  LR: {args.lr} (very low for stability)")
    print(f"  Entropy: 0 (disabled)")
    print(f"  Value pretrain: {args.value_pretrain_games} games")
    print(f"  Value recalibrate: every {args.value_recalibrate_every} episodes")
    print("=" * 60)

    trainer = StableTrainerV4(
        lr=args.lr,
        batch_size=128,
        gamma=0.99,
        entropy_coef=0,
        entropy_decay=1.0,
        min_entropy=0
    )

    trainer.load_v3_opponent(args.v3_model)
    trainer.load(args.imitation_model)

    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = args.lr

    print(f"Loaded {args.imitation_model}")

    trainer.pretrain_value_head(num_games=args.value_pretrain_games, epochs=5)

    trainer.train_stable(
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        value_recalibrate_every=args.value_recalibrate_every
    )


if __name__ == "__main__":
    main()
