import { create } from 'zustand';
import type { Route } from '../types/board';

export interface Ticket {
  source: string;
  target: string;
  points: number;
}

export interface PlayerInfo {
  hand: Record<string, number>;
  trains: number;
  points: number;
  tickets: Ticket[];
  pendingTickets?: Ticket[];
  stations: number;
}

interface LegalAction {
  index: number;
  type: string;
  source1: string;
  source2: string | null;
  card1: string | null;
  card2: string | null;
  colorCount?: number;
  locoCount?: number;
}

export type GameMode = 'visualizer' | 'singleplayer' | null;

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
  mode: GameMode;

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
    mode?: string;
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
  mode: null,

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
      mode: null,
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
      mode: (state.mode as GameMode) || null,
    });
  },
}));

export { getRouteId };
export type { PlayerInfo, LegalAction };
