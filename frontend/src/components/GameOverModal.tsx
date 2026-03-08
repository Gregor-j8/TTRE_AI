import type { PlayerInfo } from '../hooks/useGameState';
import { PLAYER_COLORS } from '../types/board';

interface GameOverModalProps {
  players: PlayerInfo[];
  mode: 'visualizer' | 'singleplayer' | null;
  onNewGame: () => void;
}

export function GameOverModal({ players, mode, onNewGame }: GameOverModalProps) {
  const sortedPlayers = players
    .map((player, idx) => ({ player, idx }))
    .sort((a, b) => b.player.points - a.player.points);

  const winner = sortedPlayers[0];
  const humanPlayer = mode === 'singleplayer' ? sortedPlayers.find(p => p.idx === 0) : null;
  const didHumanWin = humanPlayer && humanPlayer.idx === winner.idx;

  return (
    <div className="absolute inset-0 bg-black/90 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-8 w-full max-w-2xl shadow-2xl border-2 border-yellow-500">
        <div className="text-center mb-6">
          {mode === 'singleplayer' ? (
            <>
              <h1 className={`text-4xl font-bold mb-2 ${didHumanWin ? 'text-green-400' : 'text-red-400'}`}>
                {didHumanWin ? '🎉 Victory!' : '😔 Defeat'}
              </h1>
              <p className="text-gray-300 text-lg">
                {didHumanWin ? 'You won the game!' : 'Better luck next time!'}
              </p>
            </>
          ) : (
            <>
              <h1 className="text-4xl font-bold mb-2 text-yellow-400">
                🏆 Game Over
              </h1>
              <p className="text-gray-300 text-lg">
                Player {winner.idx + 1} wins!
              </p>
            </>
          )}
        </div>

        <div className="mb-6">
          <h2 className="text-xl font-semibold text-white mb-4 text-center">Final Scores</h2>
          <div className="space-y-2">
            {sortedPlayers.map(({ player, idx }, rank) => {
              const isWinner = rank === 0;
              const isHuman = mode === 'singleplayer' && idx === 0;
              const playerColor = PLAYER_COLORS[idx % PLAYER_COLORS.length];

              return (
                <div
                  key={idx}
                  className={`flex items-center justify-between p-4 rounded-lg ${
                    isWinner
                      ? 'bg-yellow-500/20 border-2 border-yellow-500'
                      : 'bg-gray-700/50 border border-gray-600'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-bold text-gray-400 w-8">
                      {rank === 0 ? '🥇' : rank === 1 ? '🥈' : rank === 2 ? '🥉' : `${rank + 1}.`}
                    </span>

                    <div className="flex items-center gap-2">
                      <div
                        className="w-8 h-8 rounded-full border-2 border-white flex items-center justify-center"
                        style={{ backgroundColor: playerColor }}
                      >
                        <span className="text-white font-bold text-sm">{idx + 1}</span>
                      </div>
                      <span className="text-white font-medium">
                        {isHuman ? 'You' : `AI Player ${idx + 1}`}
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-3xl font-bold text-yellow-400">
                      {player.points}
                    </div>
                    <div className="text-xs text-gray-400">points</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {mode === 'singleplayer' && humanPlayer && (
          <div className="mb-6 bg-gray-700/30 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">Your Stats</h3>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-400">
                  {humanPlayer.player.tickets.length}
                </div>
                <div className="text-xs text-gray-400">Tickets</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-400">
                  {45 - humanPlayer.player.trains}
                </div>
                <div className="text-xs text-gray-400">Routes Claimed</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-400">
                  {humanPlayer.player.stations}
                </div>
                <div className="text-xs text-gray-400">Stations Left</div>
              </div>
            </div>
          </div>
        )}

        <button
          onClick={onNewGame}
          className="w-full py-4 rounded-lg font-bold text-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          Play Again
        </button>
      </div>
    </div>
  );
}
