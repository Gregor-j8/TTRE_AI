import type { RouteId, RouteColor } from './board';

export type CardType = RouteColor | 'Locomotive';

export interface Ticket {
  source: string;
  target: string;
  points: number;
}

export interface PlayerState {
  hand: Record<CardType, number>;
  trains: number;
  points: number;
  tickets: Ticket[];
  claimedRoutes: RouteId[];
  stations: number;
  stationsBuilt: string[];
}

export interface GameState {
  players: PlayerState[];
  currentPlayerIdx: number;
  faceUpCards: CardType[];
  drawPileCount: number;
  claimedRoutes: Map<RouteId, number>;
  finalRound: boolean;
  gameOver: boolean;
}

export type ActionType =
  | 'draw_card'
  | 'draw_wild_card'
  | 'claim_route'
  | 'draw_tickets'
  | 'keep_tickets'
  | 'build_station';

export interface Action {
  type: ActionType;
  source1?: string;
  source2?: string;
  color?: CardType;
}
