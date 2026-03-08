import type { LegalAction } from '../../hooks/useGameState';

interface CardChoiceModalProps {
  route: { source: string; target: string; carriages: number };
  options: LegalAction[];
  onChoose: (action: LegalAction) => void;
  onCancel: () => void;
}

export function CardChoiceModal({ route, options, onChoose, onCancel }: CardChoiceModalProps) {
  const formatCardChoice = (action: LegalAction) => {
    const colorName = action.card2?.replace('_cards', '') || 'unknown';
    const colorDisplay = colorName.charAt(0).toUpperCase() + colorName.slice(1);

    const colorCount = action.colorCount || 0;
    const locoCount = action.locoCount || 0;

    const parts = [];
    if (colorCount > 0) {
      parts.push(`${colorCount} ${colorDisplay}`);
    }
    if (locoCount > 0) {
      parts.push(`${locoCount} Locomotive${locoCount > 1 ? 's' : ''}`);
    }

    return parts.join(' + ') || `${route.carriages} cards`;
  };

  return (
    <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-md shadow-2xl border border-gray-700">
        <h2 className="text-2xl font-bold text-white text-center mb-2">
          Choose Cards
        </h2>
        <p className="text-gray-400 text-sm text-center mb-6">
          {route.source} → {route.target} ({route.carriages} cards)
        </p>

        <div className="space-y-2 mb-6">
          {options.map((action, idx) => (
            <button
              key={idx}
              onClick={() => onChoose(action)}
              className="w-full p-4 rounded-lg bg-gray-700 hover:bg-gray-600 border-2 border-gray-600 hover:border-blue-500 transition-all text-left"
            >
              <div className="flex items-center justify-between">
                <div className="text-white font-medium">
                  {formatCardChoice(action)}
                </div>
                <div className="text-gray-400 text-sm">
                  →
                </div>
              </div>
            </button>
          ))}
        </div>

        <button
          onClick={onCancel}
          className="w-full py-3 rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-white transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
