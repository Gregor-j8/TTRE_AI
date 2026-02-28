import type { Route as RouteType, RouteColor, Waypoints } from '../../types/board';
import { ROUTE_COLORS } from '../../types/board';

interface RouteProps {
  route: RouteType;
  waypoints: [number, number][];
  claimedBy?: number;
}

function waypointsToPath(points: [number, number][]): string {
  if (points.length < 2) return '';
  const [first, ...rest] = points;
  return `M ${first[0]} ${first[1]} ${rest.map(([x, y]) => `L ${x} ${y}`).join(' ')}`;
}

function getRouteColor(color: RouteColor): string {
  return ROUTE_COLORS[color] || ROUTE_COLORS.Gray;
}

export function Route({ route, waypoints, claimedBy }: RouteProps) {
  const pathD = waypointsToPath(waypoints);
  const color = getRouteColor(route.color);
  const isTunnel = route.tunnel;
  const isFerry = route.engine > 0;

  return (
    <g className="route" data-route={`${route.source}-${route.target}`}>
      <path
        d={pathD}
        fill="none"
        stroke={claimedBy !== undefined ? color : 'rgba(0,0,0,0.15)'}
        strokeWidth={claimedBy !== undefined ? 6 : 4}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray={isTunnel ? '8,4' : isFerry ? '4,2,1,2' : undefined}
        style={{
          filter: claimedBy !== undefined
            ? 'drop-shadow(1px 1px 2px rgba(0,0,0,0.4))'
            : undefined,
        }}
      />
      {claimedBy === undefined && (
        <path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeDasharray={isTunnel ? '8,4' : isFerry ? '4,2,1,2' : undefined}
          opacity={0.6}
          className="hover:opacity-100 transition-opacity cursor-pointer"
        />
      )}
      <text
        x={waypoints[Math.floor(waypoints.length / 2)]?.[0] || 0}
        y={(waypoints[Math.floor(waypoints.length / 2)]?.[1] || 0) - 8}
        fontSize={7}
        fill="#666"
        textAnchor="middle"
      >
        {route.carriages}
      </text>
    </g>
  );
}
