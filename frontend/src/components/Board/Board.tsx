import { useEffect, useState } from 'react';
import type { City, Route as RouteType, Waypoints } from '../../types/board';
import { EuropeMap } from './EuropeMap';
import { CityMarker } from './CityMarker';
import { Route } from './Route';

import citiesData from '../../data/cities.csv?raw';
import routesData from '../../data/routes.csv?raw';
import waypointsData from '../../data/route_waypoints.json';

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
    routeCount[key] = (routeCount[key] || 0);
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

function getWaypointsForRoute(
  route: RouteType,
  waypoints: Waypoints
): [number, number][] {
  const key = `${route.source}-${route.target}-${route.color === 'Gray' ? 'false' : route.color}`;
  if (waypoints[key]) return waypoints[key];

  const reverseKey = `${route.target}-${route.source}-${route.color === 'Gray' ? 'false' : route.color}`;
  if (waypoints[reverseKey]) return [...waypoints[reverseKey]].reverse();

  return [];
}

export function Board() {
  const [cities, setCities] = useState<City[]>([]);
  const [routes, setRoutes] = useState<RouteType[]>([]);
  const [waypoints] = useState<Waypoints>(waypointsData as Waypoints);

  useEffect(() => {
    setCities(parseCitiesCSV(citiesData));
    setRoutes(parseRoutesCSV(routesData));
  }, []);

  return (
    <div className="w-full h-full flex items-center justify-center bg-gray-800 p-4">
      <svg
        viewBox="0 0 800 541"
        className="max-w-full max-h-full"
        style={{ aspectRatio: '800/541' }}
      >
        <EuropeMap />

        <g id="routes">
          {routes.map((route, idx) => {
            const routeWaypoints = getWaypointsForRoute(route, waypoints);
            if (routeWaypoints.length === 0) return null;
            return (
              <Route
                key={`${route.source}-${route.target}-${route.color}-${idx}`}
                route={route}
                waypoints={routeWaypoints}
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
  );
}
