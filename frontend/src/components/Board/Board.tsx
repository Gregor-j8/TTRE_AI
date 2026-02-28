import { useState } from 'react';
import type { City, Route as RouteType } from '../../types/board';
import { PLAYER_COLORS } from '../../types/board';
import { EuropeMap } from './EuropeMap';
import { CityMarker } from './CityMarker';
import { Route } from './Route';
import { useGameState, getRouteId } from '../../hooks/useGameState';
import { useWebSocket } from '../../hooks/useWebSocket';

import citiesData from '@data/cities.csv?raw';
import routesData from '@data/routes.csv?raw';

function parseCitiesCSV(csv: string): City[] {
  const lines = csv.trim().split('\n');
  return lines.slice(1).map((line) => {
    const [name, x, y] = line.split(',');
    return { name: name.trim(), x: parseFloat(x), y: parseFloat(y) };
  });
}

function parseRoutesCSV(csv: string): RouteType[] {
  const lines = csv.trim().split('\n');
  const routeCount: Record<string, number> = {};

  return lines.slice(1).map((line) => {
    const [source, target, carriages, color, tunnel, engine] = line.split(',');
    const key = `${source}-${target}`;
    routeCount[key] = routeCount[key] || 0;
    const routeKey = routeCount[key];
    routeCount[key]++;

    return {
      source: source.trim(),
      target: target.trim(),
      carriages: parseInt(carriages),
      color: color === 'false' ? 'Gray' : (color.trim() as RouteType['color']),
      tunnel: tunnel.trim() === 'true',
      engine: parseInt(engine),
      key: routeKey,
    };
  });
}

function generateWaypointsFromCities(
  route: RouteType,
  cities: City[],
  allRoutes: RouteType[]
): [number, number][] {
  const sourceCity = cities.find(c => c.name === route.source);
  const targetCity = cities.find(c => c.name === route.target);

  if (!sourceCity || !targetCity) return [];

  const isDoubleRoute = allRoutes.some(
    r => r !== route &&
    ((r.source === route.source && r.target === route.target) ||
     (r.source === route.target && r.target === route.source))
  );

  const dx = targetCity.x - sourceCity.x;
  const dy = targetCity.y - sourceCity.y;
  const len = Math.sqrt(dx * dx + dy * dy);

  const perpX = -dy / len;
  const perpY = dx / len;
  const parallelOffset = isDoubleRoute ? (route.key === 0 ? -5 : 5) : 0;

  const startX = sourceCity.x + perpX * parallelOffset;
  const startY = sourceCity.y + perpY * parallelOffset;
  const endX = targetCity.x + perpX * parallelOffset;
  const endY = targetCity.y + perpY * parallelOffset;

  if (isDoubleRoute) {
    return [
      [startX, startY],
      [endX, endY],
    ];
  }

  const curveAmount = len > 150 ? 20 : len > 80 ? 12 : 0;

  if (curveAmount === 0) {
    return [
      [startX, startY],
      [endX, endY],
    ];
  }

  const midX = (startX + endX) / 2 + perpX * curveAmount;
  const midY = (startY + endY) / 2 + perpY * curveAmount;

  return [
    [startX, startY],
    [midX, midY],
    [endX, endY],
  ];
}

const parsedCities = parseCitiesCSV(citiesData);
const parsedRoutes = parseRoutesCSV(routesData);

export function Board() {
  const [cities] = useState<City[]>(parsedCities);
  const [routes] = useState<RouteType[]>(parsedRoutes);

  const { currentPlayerIdx, claimedRoutes, claimRoute, resetGame, playerCount, players, gameOver, connectedToServer } =
    useGameState();
  const { connected, startGame } = useWebSocket();

  return (
    <div className="w-full h-full flex flex-col bg-gray-800 relative">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                connected ? 'bg-green-500' : connectedToServer ? 'bg-yellow-500' : 'bg-red-500'
              }`}
            />
            <span className="text-gray-400 text-xs">
              {connected ? 'Connected' : connectedToServer ? 'Synced' : 'Offline'}
            </span>
          </div>

          {connected && !gameOver && (
            <div className="flex gap-2">
              <button
                onClick={() => startGame('visualizer')}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded transition-colors"
              >
                Watch AI
              </button>
              <button
                onClick={() => startGame('singleplayer')}
                className="px-3 py-1 bg-green-600 hover:bg-green-500 text-white text-sm rounded transition-colors"
              >
                Play vs AI
              </button>
            </div>
          )}

          <span className="text-white text-sm">Current Player:</span>
          <div className="flex gap-2">
            {Array.from({ length: playerCount }).map((_, idx) => (
              <div
                key={idx}
                className={`w-6 h-6 rounded-full border-2 ${
                  idx === currentPlayerIdx ? 'border-white scale-125' : 'border-transparent'
                }`}
                style={{
                  backgroundColor: PLAYER_COLORS[idx],
                  transition: 'transform 0.2s',
                }}
              />
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4">
          {players.length > 0 && (
            <div className="flex gap-4">
              {players.map((player, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-2 px-2 py-1 rounded"
                  style={{ backgroundColor: `${PLAYER_COLORS[idx]}33` }}
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: PLAYER_COLORS[idx] }}
                  />
                  <span className="text-white text-xs">
                    {player.trains} trains | {player.points} pts
                  </span>
                </div>
              ))}
            </div>
          )}

          <span className="text-gray-400 text-sm">
            Routes: {claimedRoutes.size}
          </span>
          <button
            onClick={resetGame}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      {gameOver && (
        <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-8 rounded-lg text-center">
            <h2 className="text-2xl text-white mb-4">Game Over!</h2>
            <div className="flex flex-col gap-2 mb-6">
              {players
                .map((p, idx) => ({ ...p, idx }))
                .sort((a, b) => b.points - a.points)
                .map((player, rank) => (
                  <div
                    key={player.idx}
                    className="flex items-center gap-3 px-4 py-2 rounded"
                    style={{ backgroundColor: `${PLAYER_COLORS[player.idx]}44` }}
                  >
                    <span className="text-white font-bold">#{rank + 1}</span>
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: PLAYER_COLORS[player.idx] }}
                    />
                    <span className="text-white">{player.points} points</span>
                  </div>
                ))}
            </div>
            <button
              onClick={resetGame}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors"
            >
              Play Again
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 flex items-center justify-center p-4">
        <svg
          viewBox="0 0 800 541"
          className="max-h-full"
          style={{ aspectRatio: '800/541', maxWidth: '1200px' }}
        >
          <EuropeMap />

          <g id="routes">
            {routes.map((route, idx) => {
              const routeWaypoints = generateWaypointsFromCities(route, cities, routes);
              if (routeWaypoints.length === 0) return null;
              const routeId = getRouteId(route);
              const claimedBy = claimedRoutes.get(routeId);

              return (
                <Route
                  key={`${route.source}-${route.target}-${route.color}-${idx}`}
                  route={route}
                  waypoints={routeWaypoints}
                  claimedBy={claimedBy}
                  onClaim={claimRoute}
                />
              );
            })}
          </g>

          <g id="cities">
            {cities.map((city) => (
              <CityMarker key={city.name} city={city} />
            ))}
          </g>
        </svg>
      </div>
    </div>
  );
}
