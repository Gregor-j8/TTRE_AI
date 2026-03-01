export interface City {
  name: string;
  x: number;
  y: number;
}

export type RouteColor =
  | 'Red'
  | 'Blue'
  | 'Green'
  | 'Yellow'
  | 'Black'
  | 'White'
  | 'Orange'
  | 'Pink'
  | 'Gray';

export interface Route {
  source: string;
  target: string;
  carriages: number;
  color: RouteColor;
  tunnel: boolean;
  engine: number;
  key: number;
}

export type RouteId = string;

export type Waypoints = Record<string, [number, number][]>;

export const ROUTE_COLORS: Record<RouteColor, string> = {
  Red: '#e74c3c',
  Blue: '#3498db',
  Green: '#2ecc71',
  Yellow: '#f1c40f',
  Black: '#2c3e50',
  White: '#ecf0f1',
  Orange: '#e67e22',
  Pink: '#ff69b4',
  Gray: '#95a5a6',
};

export const PLAYER_COLORS = [
  '#e74c3c',
  '#3498db',
  '#2ecc71',
  '#f39c12',
  '#9b59b6',
];
