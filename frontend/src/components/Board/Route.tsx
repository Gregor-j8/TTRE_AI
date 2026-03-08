import { useState, useEffect, useRef } from 'react';
import type { Route as RouteType, RouteColor } from '../../types/board';
import { ROUTE_COLORS, PLAYER_COLORS } from '../../types/board';

interface RouteProps {
  route: RouteType;
  waypoints: [number, number][];
  claimedBy?: number;
  onClaim?: (route: RouteType) => void;
  highlightPlayer?: number | null;
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

  const compressionFactor = 0.9;
  const usableLength = totalLength * compressionFactor;
  const startOffset = (totalLength - usableLength) / 2;
  const spacing = usableLength / (count + 1);
  const result: { x: number; y: number; angle: number }[] = [];

  let currentDist = startOffset + spacing;
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

export function Route({ route, waypoints, claimedBy, onClaim, highlightPlayer }: RouteProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isNewlyClaimed, setIsNewlyClaimed] = useState(false);
  const prevClaimedRef = useRef<number | undefined>(undefined);

  const color = getRouteColor(route.color);
  const isTunnel = route.tunnel;
  const isFerry = route.engine > 0;
  const isClaimed = claimedBy !== undefined;
  const isHighlighted = highlightPlayer !== null && claimedBy === highlightPlayer;

  useEffect(() => {
    if (claimedBy !== undefined && prevClaimedRef.current === undefined) {
      setIsNewlyClaimed(true);
      const timer = setTimeout(() => setIsNewlyClaimed(false), 800);
      return () => clearTimeout(timer);
    }
    prevClaimedRef.current = claimedBy;
  }, [claimedBy]);

  const slotPositions = interpolatePoints(waypoints, route.carriages);
  const midIdx = Math.floor(slotPositions.length / 2);

  const handleClick = () => {
    if (!isClaimed && onClaim) {
      onClaim(route);
    }
  };

  const playerColor = isClaimed ? PLAYER_COLORS[claimedBy % PLAYER_COLORS.length] : color;

  return (
    <g
      className="route"
      data-route={`${route.source}-${route.target}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
      style={{ cursor: isClaimed ? 'default' : 'pointer' }}
    >
      {isHighlighted && slotPositions.map((pos, idx) => (
        <rect
          key={`glow-${idx}`}
          x={pos.x - 11}
          y={pos.y - 6}
          width={22}
          height={12}
          rx={3}
          fill="none"
          stroke={playerColor}
          strokeWidth={3}
          opacity={0.6}
          transform={`rotate(${pos.angle}, ${pos.x}, ${pos.y})`}
          style={{ filter: 'blur(2px)' }}
        />
      ))}

      {slotPositions.map((pos, idx) => {
        const isLocoSlot = isFerry && idx < route.engine;

        return (
          <g key={idx}>
            <rect
              x={pos.x - 9}
              y={pos.y - 4}
              width={18}
              height={8}
              rx={isClaimed ? 3 : 2}
              fill={playerColor}
              stroke={isClaimed ? '#111' : '#000'}
              strokeWidth={isClaimed ? 1.5 : 1}
              strokeDasharray={isTunnel && !isClaimed ? '3,2' : undefined}
              opacity={isClaimed ? 1 : isHovered ? 0.85 : 0.5}
              transform={`rotate(${pos.angle}, ${pos.x}, ${pos.y})`}
              style={{
                transition: 'opacity 0.15s, transform 0.3s',
                filter: isClaimed
                  ? `drop-shadow(1px 1px 2px rgba(0,0,0,0.5))${isNewlyClaimed ? ' brightness(1.3)' : ''}`
                  : isHovered
                  ? 'drop-shadow(0px 0px 3px rgba(0,0,0,0.4))'
                  : undefined,
              }}
            />

            {isClaimed && (
              <>
                <circle
                  cx={pos.x - 5}
                  cy={pos.y + 4}
                  r={2}
                  fill="#333"
                  stroke="#111"
                  strokeWidth={0.5}
                  transform={`rotate(${pos.angle}, ${pos.x}, ${pos.y})`}
                />
                <circle
                  cx={pos.x + 5}
                  cy={pos.y + 4}
                  r={2}
                  fill="#333"
                  stroke="#111"
                  strokeWidth={0.5}
                  transform={`rotate(${pos.angle}, ${pos.x}, ${pos.y})`}
                />
              </>
            )}

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

      {isClaimed && slotPositions.length > 0 && (
        <g pointerEvents="none">
          <circle
            cx={slotPositions[midIdx].x}
            cy={slotPositions[midIdx].y - 8}
            r={4}
            fill={playerColor}
            stroke="#fff"
            strokeWidth={1.5}
            style={{
              filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.4))',
            }}
          />
          <text
            x={slotPositions[midIdx].x}
            y={slotPositions[midIdx].y - 5.5}
            fontSize={5}
            fill="#fff"
            textAnchor="middle"
            fontWeight="bold"
          >
            {(claimedBy % PLAYER_COLORS.length) + 1}
          </text>
        </g>
      )}

      {isNewlyClaimed && slotPositions.length > 0 && (
        <circle
          cx={slotPositions[midIdx].x}
          cy={slotPositions[midIdx].y}
          r={20}
          fill="none"
          stroke={playerColor}
          strokeWidth={2}
          opacity={0}
          style={{
            animation: 'pulse-ring 0.8s ease-out',
          }}
        />
      )}

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

      <style>
        {`
          @keyframes pulse-ring {
            0% { r: 5; opacity: 0.8; }
            100% { r: 25; opacity: 0; }
          }
        `}
      </style>
    </g>
  );
}
