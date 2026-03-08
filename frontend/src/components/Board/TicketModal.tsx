import { useState } from 'react';
import type { Ticket } from '../../hooks/useGameState';

interface TicketModalProps {
  tickets: Ticket[];
  onKeep: (indices: number[]) => void;
  onCancel?: () => void;
}

export function TicketModal({ tickets, onKeep, onCancel }: TicketModalProps) {
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set([0]));

  const toggleTicket = (index: number) => {
    const newSelected = new Set(selectedIndices);
    if (newSelected.has(index)) {
      if (newSelected.size > 1) {
        newSelected.delete(index);
      }
    } else {
      newSelected.add(index);
    }
    setSelectedIndices(newSelected);
  };

  const handleKeep = () => {
    onKeep(Array.from(selectedIndices).sort());
  };

  return (
    <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 w-full max-w-lg shadow-2xl border border-gray-700">
        <h2 className="text-2xl font-bold text-white text-center mb-4">
          Choose Destination Tickets
        </h2>
        <p className="text-gray-400 text-sm text-center mb-6">
          Select at least 1 ticket to keep (you can keep all 3)
        </p>

        <div className="space-y-3 mb-6">
          {tickets.map((ticket, idx) => (
            <div
              key={idx}
              onClick={() => toggleTicket(idx)}
              className={`p-4 rounded-lg cursor-pointer transition-all border-2 ${
                selectedIndices.has(idx)
                  ? 'bg-blue-600/20 border-blue-500'
                  : 'bg-gray-700/50 border-gray-600 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="text-white font-medium text-lg">
                    {ticket.source} → {ticket.target}
                  </div>
                  <div className="text-gray-400 text-sm mt-1">
                    Connect these cities to earn points
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-2xl font-bold text-yellow-400">
                    {ticket.points}
                  </div>
                  <div className="w-6 h-6 rounded border-2 flex items-center justify-center">
                    {selectedIndices.has(idx) && (
                      <div className="w-4 h-4 rounded bg-blue-500" />
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-3">
          {onCancel && (
            <button
              onClick={onCancel}
              className="flex-1 py-3 rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            >
              Cancel
            </button>
          )}
          <button
            onClick={handleKeep}
            disabled={selectedIndices.size === 0}
            className={`flex-1 py-3 rounded-lg font-bold transition-colors ${
              selectedIndices.size === 0
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-500 text-white'
            }`}
          >
            Keep {selectedIndices.size} Ticket{selectedIndices.size !== 1 ? 's' : ''}
          </button>
        </div>
      </div>
    </div>
  );
}
