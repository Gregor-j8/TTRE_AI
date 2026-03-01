import type { City } from '../../types/board';

interface CityMarkerProps {
  city: City;
}

export function CityMarker({ city }: CityMarkerProps) {
  const labelOffset = 12;
  const labelY = city.y < 100 ? city.y + labelOffset : city.y - labelOffset;

  return (
    <g className="city-marker">
      <circle
        cx={city.x}
        cy={city.y}
        r={6}
        fill="#8b4513"
        stroke="#5c2d0e"
        strokeWidth={1.5}
        style={{ filter: 'drop-shadow(1px 1px 1px rgba(0,0,0,0.3))' }}
      />
      <circle cx={city.x} cy={city.y} r={3} fill="#d4a574" />
      <text
        x={city.x}
        y={labelY}
        textAnchor="middle"
        fontSize={8}
        fontFamily="Georgia, serif"
        fill="#3d2914"
        fontWeight="bold"
        style={{ filter: 'drop-shadow(0px 0px 2px rgba(255,255,255,0.8))' }}
      >
        {city.name}
      </text>
    </g>
  );
}
