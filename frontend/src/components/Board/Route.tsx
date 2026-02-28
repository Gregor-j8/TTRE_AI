import { useState } from 'react';
import type { Route as RouteType, RouteColor } from '../../types/board';
import { ROUTE_COLORS, PLAYER_COLORS } from '../../types/board';

interface RouteProps {
  route: RouteType;
  waypoints: [number, number][];
  claimedBy?: number;
  onClaim?: (route: RouteType) => void;
}

function waypointsToPath(points: [number, number][]): string {
  if (points.length < 2) return '';
  const [first, ...rest] = points;
  return `M ${first[0]} ${first[1]} ${rest.map(([x, y]) => `L ${x} ${y}`).join(' ')}`;
}

function getRouteColor(color: RouteColor): string {
  return ROUTE_COLORS[color] || ROUTE_COLORS.Gray;
}

function interpolatePoints(
  points: [number, number][],
  count: number
): { x: number; y: number; angle: number }[] {
  if (points.length < 2) return [];

  const totalLength = points.slice(1).reduce((acc, p, i) => {
    const prev = points[i];
    return acc + Math.hypot(p[0] - prev[0], p[1] - prev[1]);
  }, 0);

  const spacing = totalLength / (count + 1);
  const result: { x: number; y: number; angle: number }[] = [];

  let currentDist = spacing;
  let segmentStart = 0;
  let accumulated = 0;

  for (let i = 1; i < points.length && result.length < count; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const segLen = Math.hypot(curr[0] - prev[0], curr[1] - prev[1]);

    while (currentDist <= accumulated + segLen && result.length < count) {
      const t = (currentDist - accumulated) / segLen;
      const x = prev[0] + t * (curr[0] - prev[0]);
      const y = prev[1] + t * (curr[1] - prev[1]);
      const angle = Math.atan2(curr[1] - prev[1], curr[0] - prev[0]) * (180 / Math.PI);
      result.push({ x, y, angle });
      currentDist += spacing;
    }
    accumulated += segLen;
  }

  return result;
}

export function Route({ route, waypoints, claimedBy, onClaim }: RouteProps) {
  const [isHovered, setIsHovered] = useState(false);
  const color = getRouteColor(route.color);
  const isTunnel = route.tunnel;
  const isFerry = route.engine > 0;
  const isClaimed = claimedBy !== undefined;

  const slotPositions = interpolatePoints(waypoints, route.carriages);

  const handleClick = () => {
    if (!isClaimed && onClaim) {
      onClaim(route);
    }
  };

  return (
    <g
      className="route"
      data-route={`${route.source}-${route.target}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
      style={{ cursor: isClaimed ? 'default' : 'pointer' }}
    >
      {slotPositions.map((pos, idx) => {
        const isLocoSlot = isFerry && idx < route.engine;

        return (
          <g key={idx}>
            <rect
              x={pos.x - 9}
              y={pos.y - 4.5}
              width={18}
              height={9}
              rx={2}
              fill={isClaimed ? PLAYER_COLORS[claimedBy % PLAYER_COLORS.length] : color}
              stroke={isClaimed ? '#222' : '#000'}
              strokeWidth={isClaimed ? 1 : 1.5}
              strokeDasharray={isTunnel && !isClaimed ? '3,2' : undefined}
              opacity={isClaimed ? 1 : isHovered ? 0.9 : 0.7}
              transform={`rotate(${pos.angle}, ${pos.x}, ${pos.y})`}
              style={{
                transition: 'opacity 0.15s',
                filter: isClaimed
                  ? 'drop-shadow(1px 1px 2px rgba(0,0,0,0.4))'
                  : isHovered
                  ? 'drop-shadow(0px 0px 2px rgba(0,0,0,0.3))'
                  : undefined,
              }}
            />
            {isLocoSlot && !isClaimed && (
              <text
                x={pos.x}
                y={pos.y + 3}
                fontSize={8}
                fill="#000"
                textAnchor="middle"
                fontWeight="bold"
                transform={`rotate(${pos.angle}, ${pos.x}, ${pos.y})`}
                pointerEvents="none"
              >
                🚂
              </text>
            )}
          </g>
        );
      })}

      {isHovered && !isClaimed && (
        <g pointerEvents="none">
          <rect
            x={waypoints[0][0] - 30}
            y={waypoints[0][1] - 25}
            width={60}
            height={18}
            rx={3}
            fill="rgba(0,0,0,0.85)"
          />
          <text
            x={waypoints[0][0]}
            y={waypoints[0][1] - 12}
            fontSize={9}
            fill="white"
            textAnchor="middle"
            fontFamily="sans-serif"
          >
            {route.carriages} {route.color === 'Gray' ? 'Any' : route.color}
            {isTunnel ? ' ⛰️' : ''}
            {isFerry ? ` +${route.engine}🚂` : ''}
          </text>
        </g>
      )}
    </g>
  );
}
