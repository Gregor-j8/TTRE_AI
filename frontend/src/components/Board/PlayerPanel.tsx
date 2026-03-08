import { PLAYER_COLORS } from '../../types/board';
import type { PlayerInfo } from '../../hooks/useGameState';
import type { Route } from '../../types/board';
import { areCitiesConnected } from '../../utils/pathfinding';

interface PlayerPanelProps {
  player: PlayerInfo;
  playerIdx: number;
  isCurrentPlayer: boolean;
  claimedRoutes?: Route[];
}

const CARD_KEYS = [
  'red_cards', 'blue_cards', 'green_cards', 'yellow_cards',
  'black_cards', 'white_cards', 'orange_cards', 'pink_cards', 'Locomotive'
] as const;

const CARD_COLORS: Record<string, string> = {
  red_cards: '#dc2626',
  blue_cards: '#2563eb',
  green_cards: '#16a34a',
  yellow_cards: '#eab308',
  black_cards: '#1f2937',
  white_cards: '#f3f4f6',
  orange_cards: '#ea580c',
  pink_cards: '#ec4899',
  Locomotive: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
};

const CARD_NAMES: Record<string, string> = {
  red_cards: 'R',
  blue_cards: 'B',
  green_cards: 'G',
  yellow_cards: 'Y',
  black_cards: 'K',
  white_cards: 'W',
  orange_cards: 'O',
  pink_cards: 'P',
  Locomotive: '🚂',
};

function MiniCard({ cardKey, count }: { cardKey: string; count: number }) {
  if (count === 0) return null;

  const bgColor = CARD_COLORS[cardKey] || '#666';
  const label = CARD_NAMES[cardKey] || '?';
  const isLightColor = cardKey === 'white_cards' || cardKey === 'yellow_cards';

  return (
    <div
      className="flex items-center justify-center rounded w-9 h-7 text-xs font-bold shadow"
      style={{
        background: bgColor,
        color: isLightColor ? '#333' : '#fff',
        border: isLightColor ? '1px solid #9ca3af' : '1px solid rgba(0,0,0,0.2)',
      }}
    >
      <span>{count}{label}</span>
    </div>
  );
}

export function PlayerPanel({ player, playerIdx, isCurrentPlayer, claimedRoutes = [] }: PlayerPanelProps) {
  return (
    <div
      className={`bg-gray-800/90 rounded-lg p-3 w-80 ${
        isCurrentPlayer ? 'ring-2 ring-white/50' : ''
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-4 h-4 rounded-full shrink-0"
          style={{ backgroundColor: PLAYER_COLORS[playerIdx] }}
        />
        <span className="text-white text-sm font-medium">P{playerIdx + 1}</span>
        <span className="text-gray-400 text-xs">
          {player.trains}🚂 {player.points}pts
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5 min-h-8">
        {CARD_KEYS.map((cardKey) => (
          <MiniCard
            key={cardKey}
            cardKey={cardKey}
            count={player.hand[cardKey] || 0}
          />
        ))}
      </div>

      {player.tickets.length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-700">
          <div className="flex flex-wrap gap-1">
            {player.tickets.map((ticket, idx) => {
              const isCompleted = areCitiesConnected(ticket.source, ticket.target, claimedRoutes);
              return (
                <div
                  key={idx}
                  className={`rounded px-1.5 py-0.5 text-[10px] flex items-center gap-1 ${
                    isCompleted
                      ? 'bg-green-600/30 border border-green-500/50 text-green-200'
                      : 'bg-gray-700 text-gray-300'
                  }`}
                >
                  {isCompleted && <span className="text-green-400">✓</span>}
                  {ticket.source}→{ticket.target}
                  <span className="text-yellow-400 ml-0.5">{ticket.points}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
