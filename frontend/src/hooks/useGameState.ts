import { create } from 'zustand';
import type { Route } from '../types/board';

interface PlayerInfo {
  hand: Record<string, number>;
  trains: number;
  points: number;
  tickets: unknown[];
  stations: number;
}

interface LegalAction {
  index: number;
  type: string;
  source1: string;
  source2: string | null;
  card1: string | null;
  card2: string | null;
}

interface GameState {
  currentPlayerIdx: number;
  playerCount: number;
  claimedRoutes: Map<string, number>;
  players: PlayerInfo[];
  faceUpCards: string[];
  gameOver: boolean;
  finalRound: boolean;
  legalActions: LegalAction[];
  connectedToServer: boolean;

  claimRoute: (route: Route) => void;
  resetGame: () => void;
  nextPlayer: () => void;
  setStateFromServer: (state: {
    currentPlayerIdx: number;
    claimedRoutes: Record<string, number>;
    players: PlayerInfo[];
    faceUpCards: string[];
    gameOver: boolean;
    finalRound: boolean;
    legalActions: LegalAction[];
  }) => void;
}

function getRouteId(route: Route): string {
  return `${route.source}-${route.target}-${route.key}`;
}

export const useGameState = create<GameState>((set, get) => ({
  currentPlayerIdx: 0,
  playerCount: 2,
  claimedRoutes: new Map(),
  players: [],
  faceUpCards: [],
  gameOver: false,
  finalRound: false,
  legalActions: [],
  connectedToServer: false,

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
      players: [],
      faceUpCards: [],
      gameOver: false,
      finalRound: false,
      legalActions: [],
    });
  },

  nextPlayer: () => {
    const { currentPlayerIdx, playerCount } = get();
    set({ currentPlayerIdx: (currentPlayerIdx + 1) % playerCount });
  },

  setStateFromServer: (state) => {
    const claimedMap = new Map<string, number>();
    for (const [routeId, playerIdx] of Object.entries(state.claimedRoutes)) {
      claimedMap.set(routeId, playerIdx);
    }

    set({
      currentPlayerIdx: state.currentPlayerIdx,
      claimedRoutes: claimedMap,
      players: state.players,
      faceUpCards: state.faceUpCards,
      gameOver: state.gameOver,
      finalRound: state.finalRound,
      legalActions: state.legalActions || [],
      connectedToServer: true,
      playerCount: state.players.length || 2,
    });
  },
}));

export { getRouteId };
export type { PlayerInfo, LegalAction };
