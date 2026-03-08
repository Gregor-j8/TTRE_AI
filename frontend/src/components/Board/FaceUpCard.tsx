interface FaceUpCardProps {
  color: string;
  clickable?: boolean;
  selected?: boolean;
  onClick?: () => void;
}

const CARD_COLORS: Record<string, string> = {
  red_cards: '#dc2626',
  blue_cards: '#2563eb',
  green_cards: '#16a34a',
  yellow_cards: '#eab308',
  black_cards: '#1f2937',
  white_cards: '#f3f4f6',
  orange_cards: '#ea580c',
  pink_cards: '#ec4899',
};

const DISPLAY_NAMES: Record<string, string> = {
  red_cards: 'Red',
  blue_cards: 'Blue',
  green_cards: 'Green',
  yellow_cards: 'Yellow',
  black_cards: 'Black',
  white_cards: 'White',
  orange_cards: 'Orange',
  pink_cards: 'Pink',
  Locomotive: 'Wild',
};

export function FaceUpCard({ color, clickable, selected, onClick }: FaceUpCardProps) {
  const isLocomotive = color === 'Locomotive';

  const bgColor = isLocomotive
    ? 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #8b5cf6 100%)'
    : CARD_COLORS[color] || '#666';

  const isLightColor = color === 'white_cards' || color === 'yellow_cards';
  const textColor = isLightColor ? '#1f2937' : '#fff';
  const displayName = DISPLAY_NAMES[color] || color;

  return (
    <div
      onClick={clickable || selected ? onClick : undefined}
      className={`relative flex flex-col items-center justify-center rounded-lg w-14 h-20 shadow-xl transition-all ${
        selected
          ? 'cursor-pointer scale-110 -translate-y-2 ring-2 ring-yellow-400 ring-offset-2 ring-offset-gray-800'
          : clickable
          ? 'cursor-pointer hover:scale-110 hover:-translate-y-2 ring-2 ring-green-400 ring-offset-2 ring-offset-gray-800'
          : 'cursor-default opacity-60'
      }`}
      style={{
        background: bgColor,
        border: isLightColor ? '2px solid #9ca3af' : '2px solid rgba(0,0,0,0.2)',
        boxShadow: selected
          ? '0 4px 16px rgba(250, 204, 21, 0.5)'
          : clickable
          ? '0 4px 12px rgba(74, 222, 128, 0.4)'
          : '0 4px 12px rgba(0,0,0,0.3)',
      }}
    >
      <div className="text-2xl mb-1" style={{ color: textColor, opacity: 0.9 }}>
        {isLocomotive ? '🚂' : '🚃'}
      </div>
      <span className="text-[10px] font-bold" style={{ color: textColor }}>
        {displayName}
      </span>
    </div>
  );
}
