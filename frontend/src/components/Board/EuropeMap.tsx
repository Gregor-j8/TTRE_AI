export function EuropeMap() {
  return (
    <g id="map-background">
      <defs>
        <filter id="vintage-filter">
          <feColorMatrix
            type="matrix"
            values="0.95 0.05 0.05 0 0.02
                    0.05 0.9  0.05 0 0.02
                    0.02 0.02 0.75 0 0.02
                    0    0    0    1 0"
          />
        </filter>
      </defs>

      <image
        href="/TTRE_map.png"
        x="0"
        y="0"
        width="800"
        height="541"
        preserveAspectRatio="none"
      />
    </g>
  );
}
