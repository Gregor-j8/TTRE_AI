import { useEffect, useState } from 'react';
import type { City, Route as RouteType } from '../../types/board';
import { PLAYER_COLORS } from '../../types/board';
import { EuropeMap } from './EuropeMap';
import { CityMarker } from './CityMarker';
import { Route } from './Route';
import { useGameState, getRouteId } from '../../hooks/useGameState';

import citiesData from '../../data/cities.csv?raw';
import routesData from '../../data/routes.csv?raw';

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
  cities: City[]
): [number, number][] {
  const sourceCity = cities.find(c => c.name === route.source);
  const targetCity = cities.find(c => c.name === route.target);

  if (!sourceCity || !targetCity) return [];

  const dx = targetCity.x - sourceCity.x;
  const dy = targetCity.y - sourceCity.y;
  const len = Math.sqrt(dx * dx + dy * dy);

  const perpX = -dy / len;
  const perpY = dx / len;
  const offset = route.key * 8;

  return [
    [sourceCity.x + perpX * offset, sourceCity.y + perpY * offset],
    [targetCity.x + perpX * offset, targetCity.y + perpY * offset],
  ];
}

export function Board() {
  const [cities, setCities] = useState<City[]>([]);
  const [routes, setRoutes] = useState<RouteType[]>([]);

  const { currentPlayerIdx, claimedRoutes, claimRoute, resetGame, playerCount } =
    useGameState();

  useEffect(() => {
    setCities(parseCitiesCSV(citiesData));
    setRoutes(parseRoutesCSV(routesData));
  }, []);

  return (
    <div className="w-full h-full flex flex-col bg-gray-800">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900">
        <div className="flex items-center gap-4">
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
          <span className="text-gray-400 text-sm">
            Routes claimed: {claimedRoutes.size}
          </span>
          <button
            onClick={resetGame}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center p-4">
        <svg
          viewBox="0 0 800 541"
          className="max-h-full"
          style={{ aspectRatio: '800/541', maxWidth: '1200px' }}
        >
          <EuropeMap />

          <g id="routes">
            {routes.map((route, idx) => {
              const routeWaypoints = generateWaypointsFromCities(route, cities);
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
