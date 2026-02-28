import { create } from 'zustand';
import type { Route } from '../types/board';

interface ClaimedRoute {
  routeId: string;
  playerIdx: number;
}

interface GameState {
  currentPlayerIdx: number;
  playerCount: number;
  claimedRoutes: Map<string, number>;

  claimRoute: (route: Route) => void;
  resetGame: () => void;
  nextPlayer: () => void;
}

function getRouteId(route: Route): string {
  return `${route.source}-${route.target}-${route.color}-${route.key}`;
}

export const useGameState = create<GameState>((set, get) => ({
  currentPlayerIdx: 0,
  playerCount: 4,
  claimedRoutes: new Map(),

  claimRoute: (route: Route) => {
    const routeId = getRouteId(route);
    const { claimedRoutes, currentPlayerIdx, playerCount } = get();

    if (claimedRoutes.has(routeId)) return;

    const newClaimed = new Map(claimedRoutes);
    newClaimed.set(routeId, currentPlayerIdx);

    set({
      claimedRoutes: newClaimed,
      currentPlayerIdx: (currentPlayerIdx + 1) % playerCount,
    });
  },

  resetGame: () => {
    set({
      currentPlayerIdx: 0,
      claimedRoutes: new Map(),
    });
  },

  nextPlayer: () => {
    const { currentPlayerIdx, playerCount } = get();
    set({ currentPlayerIdx: (currentPlayerIdx + 1) % playerCount });
  },
}));

export { getRouteId };
