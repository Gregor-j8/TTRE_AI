import { useState } from 'react';
import { PLAYER_COLORS } from '../types/board';

interface StartMatchModalProps {
  onStartGame: (mode: 'visualizer' | 'singleplayer', playerCount: number, aiTypes: string[]) => void;
}

const AI_OPTIONS = [
  { value: 'random', label: 'Random', description: 'Pure random moves', category: 'heuristic' },
  { value: 'greedy', label: 'Greedy', description: 'Maximizes route points', category: 'heuristic' },
  { value: 'ticket_focused', label: 'Ticket Focused', description: 'Completes destination tickets', category: 'heuristic' },
  { value: 'smart_ticket', label: 'Smart Ticket', description: 'Efficient ticket selection', category: 'heuristic' },
  { value: 'overall_game', label: 'Overall Game', description: 'Phase-aware strategy', category: 'heuristic' },
  { value: 'blitz', label: 'Blitz', description: 'Aggressive endpoint focus', category: 'heuristic' },
  { value: 'v5_best', label: 'ML v5 Best', description: 'Neural network (best checkpoint)', category: 'ml' },
  { value: 'v5_final', label: 'ML v5 Final', description: 'Neural network (final training)', category: 'ml' },
  { value: 'v5_mixed', label: 'ML v5 Mixed', description: 'Neural network (mixed training)', category: 'ml' },
  { value: 'v4_best', label: 'ML v4 Best', description: 'Neural network v4 (64% vs heuristic)', category: 'ml' },
  { value: 'v3_final', label: 'ML v3 Final', description: 'Neural network v3', category: 'ml' },
  { value: 'v1_best', label: 'ML v1 Best', description: 'Neural network v1 (100% vs heuristic)', category: 'ml' },
];

export function StartMatchModal({ onStartGame }: StartMatchModalProps) {
  const [playerCount, setPlayerCount] = useState(2);
  const [aiTypes, setAiTypes] = useState<string[]>(['ticket_focused', 'ticket_focused', 'ticket_focused', 'ticket_focused']);
  const [isWatchMode, setIsWatchMode] = useState(false);

  const handleAiChange = (index: number, value: string) => {
    const newTypes = [...aiTypes];
    newTypes[index] = value;
    setAiTypes(newTypes);
  };

  const handleStart = (mode: 'visualizer' | 'singleplayer') => {
    const aiCount = mode === 'visualizer' ? playerCount : playerCount - 1;
    onStartGame(mode, playerCount, aiTypes.slice(0, aiCount));
  };

  const aiSlots = isWatchMode ? playerCount : playerCount - 1;

  return (
    <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md shadow-2xl border border-gray-700">
        <h2 className="text-2xl font-bold text-white text-center mb-6">
          Start New Game
        </h2>

        <div className="mb-6">
          <label className="text-gray-400 text-sm mb-2 block">Number of Players</label>
          <div className="flex gap-2">
            {[2, 3, 4, 5].map((count) => (
              <button
                key={count}
                onClick={() => setPlayerCount(count)}
                className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                  playerCount === count
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {count}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-6">
          <label className="text-gray-400 text-sm mb-2 block">Game Mode</label>
          <div className="flex gap-2">
            <button
              onClick={() => setIsWatchMode(false)}
              className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                !isWatchMode
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Play vs AI
            </button>
            <button
              onClick={() => setIsWatchMode(true)}
              className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                isWatchMode
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Watch AI Battle
            </button>
          </div>
        </div>

        <div className="mb-6 space-y-3">
          <label className="text-gray-400 text-sm block">Players</label>

          {!isWatchMode && (
            <div className="flex items-center gap-3 p-3 rounded-lg bg-gray-700/50">
              <div
                className="w-6 h-6 rounded-full flex-shrink-0"
                style={{ backgroundColor: PLAYER_COLORS[0] }}
              />
              <div className="flex-1">
                <span className="text-white font-medium">Player 1</span>
                <span className="text-green-400 text-sm ml-2">YOU</span>
              </div>
            </div>
          )}

          {Array.from({ length: aiSlots }).map((_, idx) => {
            const playerIdx = isWatchMode ? idx : idx + 1;
            return (
              <div
                key={idx}
                className="flex items-center gap-3 p-3 rounded-lg bg-gray-700/50"
              >
                <div
                  className="w-6 h-6 rounded-full flex-shrink-0"
                  style={{ backgroundColor: PLAYER_COLORS[playerIdx] }}
                />
                <div className="flex-1">
                  <span className="text-white font-medium">Player {playerIdx + 1}</span>
                  <span className="text-gray-400 text-sm ml-2">AI</span>
                </div>
                <select
                  value={aiTypes[idx]}
                  onChange={(e) => handleAiChange(idx, e.target.value)}
                  className="bg-gray-600 text-white text-sm rounded-lg px-3 py-1.5 border border-gray-500 focus:outline-none focus:border-blue-500"
                >
                  {AI_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>

        <button
          onClick={() => handleStart(isWatchMode ? 'visualizer' : 'singleplayer')}
          className={`w-full py-3 rounded-lg font-bold text-lg transition-colors ${
            isWatchMode
              ? 'bg-purple-600 hover:bg-purple-500 text-white'
              : 'bg-green-600 hover:bg-green-500 text-white'
          }`}
        >
          {isWatchMode ? 'Watch AI Battle' : 'Start Game'}
        </button>
      </div>
    </div>
  );
}
