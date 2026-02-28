export function EuropeMap() {
  return (
    <g id="map-background">
      <defs>
        <pattern
          id="paper-texture"
          patternUnits="userSpaceOnUse"
          width="100"
          height="100"
        >
          <rect width="100" height="100" fill="#f4e4bc" />
          <circle cx="20" cy="30" r="1" fill="#e8d4a8" opacity="0.5" />
          <circle cx="70" cy="60" r="1.5" fill="#e8d4a8" opacity="0.4" />
          <circle cx="40" cy="80" r="1" fill="#e8d4a8" opacity="0.3" />
        </pattern>
        <linearGradient id="water-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#b8d4e8" />
          <stop offset="100%" stopColor="#a0c4d8" />
        </linearGradient>
      </defs>

      <rect
        x="0"
        y="0"
        width="800"
        height="541"
        fill="url(#paper-texture)"
      />

      <path
        id="atlantic"
        d="M 0 0 L 0 541 L 60 541 L 60 520 Q 70 480, 50 450
           Q 30 420, 40 380 L 60 350 Q 80 300, 100 280
           L 120 250 Q 140 200, 160 180 L 180 140
           Q 200 100, 180 60 L 160 0 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="north-sea"
        d="M 160 0 L 180 60 Q 200 100, 180 140
           L 200 150 Q 240 120, 280 100 L 320 80
           Q 360 60, 400 50 L 400 0 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="baltic"
        d="M 400 0 L 400 50 Q 420 60, 440 50
           Q 480 80, 520 70 Q 560 60, 600 80
           Q 640 60, 680 70 L 680 0 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="mediterranean"
        d="M 60 520 L 60 541 L 550 541 L 550 520
           Q 520 500, 490 510 Q 450 520, 420 500
           Q 380 520, 340 510 Q 300 530, 260 510
           Q 220 530, 180 510 Q 140 530, 100 510
           Q 80 530, 60 520 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="adriatic"
        d="M 340 510 Q 350 480, 360 450 Q 370 420, 380 400
           Q 400 380, 420 350 L 440 360
           Q 420 390, 400 420 Q 380 450, 370 480
           Q 360 510, 340 510 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="aegean"
        d="M 490 510 Q 500 480, 520 460 Q 540 440, 560 450
           Q 580 460, 590 490 Q 600 510, 590 541
           L 550 541 L 550 520 Q 520 500, 490 510 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="black-sea"
        d="M 560 450 Q 580 430, 620 420 Q 660 400, 700 380
           Q 740 360, 780 350 L 800 360 L 800 420
           Q 760 400, 720 420 Q 680 440, 640 450
           Q 600 460, 560 450 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="english-channel"
        d="M 120 250 L 140 240 Q 160 220, 200 200
           Q 220 190, 240 180 L 260 200
           Q 230 210, 200 220 Q 170 230, 140 250
           Q 130 260, 120 250 Z"
        fill="url(#water-gradient)"
      />

      <path
        id="british-isles"
        d="M 100 50 Q 130 40, 160 50 Q 190 60, 200 90
           Q 210 120, 200 150 Q 190 170, 170 180
           Q 150 170, 130 150 Q 110 130, 100 100
           Q 90 70, 100 50 Z"
        fill="url(#paper-texture)"
        stroke="#c9b896"
        strokeWidth="1"
      />

      <path
        id="sicily"
        d="M 380 490 Q 400 480, 420 490 Q 430 500, 420 510
           Q 400 520, 380 510 Q 370 500, 380 490 Z"
        fill="url(#paper-texture)"
        stroke="#c9b896"
        strokeWidth="0.5"
      />

      <g id="country-borders" stroke="#c9b896" strokeWidth="0.5" fill="none">
        <path d="M 200 250 Q 220 300, 280 320" />
        <path d="M 280 320 Q 320 280, 350 250" />
        <path d="M 350 250 Q 400 270, 430 280" />
        <path d="M 430 280 Q 480 260, 510 200" />
        <path d="M 510 200 Q 550 180, 600 160" />
        <path d="M 250 380 Q 300 350, 340 340" />
        <path d="M 340 340 Q 380 330, 420 350" />
      </g>
    </g>
  );
}
