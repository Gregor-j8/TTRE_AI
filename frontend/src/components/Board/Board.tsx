import { useState, useEffect } from 'react';
import type { City, Route as RouteType } from '../../types/board';
import { PLAYER_COLORS } from '../../types/board';
import { EuropeMap } from './EuropeMap';
import { CityMarker } from './CityMarker';
import { Route } from './Route';
import { PlayerPanel } from './PlayerPanel';
import { StartMatchModal } from '../StartMatchModal';
import { TicketModal } from './TicketModal';
import { CardChoiceModal } from './CardChoiceModal';
import { GameOverModal } from '../GameOverModal';
import { FaceUpCard } from './FaceUpCard';
import { useGameState, getRouteId, type LegalAction } from '../../hooks/useGameState';
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

function shouldShowPlayer(playerIdx: number, mode: 'visualizer' | 'singleplayer' | null): boolean {
  if (mode === 'singleplayer') return playerIdx === 0;
  return true;
}

const parsedCities = parseCitiesCSV(citiesData);
const parsedRoutes = parseRoutesCSV(routesData);

export function Board() {
  const [cities] = useState<City[]>(parsedCities);
  const [routes] = useState<RouteType[]>(parsedRoutes);
  const [showStartModal, setShowStartModal] = useState(true);
  const [speed, setSpeedLocal] = useState(1);
  const [hoveredPlayer, setHoveredPlayer] = useState<number | null>(null);
  const [firstPick, setFirstPick] = useState<{ type: 'deck' | 'face_up'; card?: string } | null>(null);
  const [showTicketModal, setShowTicketModal] = useState(false);
  const [cardChoiceModal, setCardChoiceModal] = useState<{ route: RouteType; actions: LegalAction[] } | null>(null);

  const { currentPlayerIdx, claimedRoutes, claimRoute, resetGame, playerCount, players, gameOver, connectedToServer, mode, faceUpCards, legalActions } =
    useGameState();
  const { connected, startGame, setSpeed, sendAction } = useWebSocket();

  const isMyTurn = mode === 'singleplayer' && currentPlayerIdx === 0;

  const handleRouteClaim = (route: RouteType) => {
    const routeId = getRouteId(route);

    if (mode === 'singleplayer' && isMyTurn) {
      const routeKeyStr = String(route.key);
      const matchingActions = legalActions.filter(action => {
        if (action.type !== 'claim_route') return false;

        const citiesMatch =
          (action.source1 === route.source && action.source2 === route.target) ||
          (action.source1 === route.target && action.source2 === route.source);

        return citiesMatch && action.card1 === routeKeyStr;
      });

      if (matchingActions.length === 1) {
        sendAction(matchingActions[0].index);
      } else if (matchingActions.length > 1) {
        setCardChoiceModal({ route, actions: matchingActions });
      }
    } else if (mode === 'visualizer') {
      claimRoute(route);
    }
  };

  useEffect(() => {
    setFirstPick(null);
  }, [currentPlayerIdx, gameOver]);

  const findMatchingAction = (pick1: { type: 'deck' | 'face_up'; card?: string }, pick2: { type: 'deck' | 'face_up'; card?: string }) => {
    return legalActions.find(action => {
      if (action.type !== 'draw_card') return false;

      const source1Match = pick1.type === 'deck'
        ? action.source1 === 'deck'
        : action.source1 === 'face_up' && action.card1 === pick1.card;

      const source2Match = pick2.type === 'deck'
        ? action.source2 === 'deck'
        : action.source2 === 'face_up' && action.card2 === pick2.card;

      return source1Match && source2Match;
    });
  };

  const canPickFirst = (type: 'deck' | 'face_up', card?: string) => {
    if (!isMyTurn || firstPick) return false;

    return legalActions.some(action => {
      if (action.type === 'draw_wild_card' && card === 'Locomotive') {
        return true;
      }
      if (action.type !== 'draw_card') return false;

      if (type === 'deck') {
        return action.source1 === 'deck';
      }
      return action.source1 === 'face_up' && action.card1 === card;
    });
  };

  const canPickSecond = (type: 'deck' | 'face_up', card?: string) => {
    if (!isMyTurn || !firstPick) return false;

    return legalActions.some(action => {
      if (action.type !== 'draw_card') return false;

      const source1Match = firstPick.type === 'deck'
        ? action.source1 === 'deck'
        : action.source1 === 'face_up' && action.card1 === firstPick.card;

      if (!source1Match) return false;

      if (type === 'deck') {
        return action.source2 === 'deck';
      }
      return action.source2 === 'face_up' && action.card2 === card;
    });
  };

  const handleCardPick = (type: 'deck' | 'face_up', card?: string) => {
    if (!isMyTurn) return;

    if (card === 'Locomotive' && type === 'face_up') {
      const wildAction = legalActions.find(a => a.type === 'draw_wild_card' && a.card1 === 'Locomotive');
      if (wildAction) {
        sendAction(wildAction.index);
        setFirstPick(null);
      }
      return;
    }

    if (!firstPick) {
      if (canPickFirst(type, card)) {
        setFirstPick({ type, card });
      }
    } else {
      if (canPickSecond(type, card)) {
        const action = findMatchingAction(firstPick, { type, card });
        if (action) {
          sendAction(action.index);
          setFirstPick(null);
        }
      }
    }
  };

  const handleSpeedChange = (newSpeed: number) => {
    setSpeedLocal(newSpeed);
    setSpeed(newSpeed);
  };

  const handleStartGame = (gameMode: 'visualizer' | 'singleplayer', count: number, aiTypes: string[]) => {
    startGame(gameMode, count, aiTypes);
    setShowStartModal(false);
  };

  const handleReset = () => {
    resetGame();
    setShowStartModal(true);
  };

  const handleDrawTickets = () => {
    const drawTicketAction = legalActions.find(a => a.type === 'draw_tickets');
    if (drawTicketAction) {
      sendAction(drawTicketAction.index);
      setShowTicketModal(true);
    }
  };

  const handleKeepTickets = (indices: number[]) => {
    const keepAction = legalActions.find(a => a.type === 'keep_tickets' && a.source1 === indices.join(','));
    if (keepAction) {
      sendAction(keepAction.index);
      setShowTicketModal(false);
    }
  };

  const myPlayer = players[0];
  const hasPendingTickets = myPlayer?.pendingTickets && myPlayer.pendingTickets.length > 0;

  useEffect(() => {
    if (hasPendingTickets && isMyTurn) {
      setShowTicketModal(true);
    }
  }, [hasPendingTickets, isMyTurn]);

  const canDrawTickets = legalActions.some(a => a.type === 'draw_tickets');

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

          {connected && !gameOver && !showStartModal && (
            <button
              onClick={() => setShowStartModal(true)}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded transition-colors"
            >
              New Game
            </button>
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

          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs">Speed:</span>
            <input
              type="range"
              min="0.25"
              max="5"
              step="0.25"
              value={speed}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
              className="w-20 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            />
            <span className="text-white text-xs w-8">{speed}x</span>
          </div>

          <button
            onClick={handleReset}
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
              onClick={handleReset}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors"
            >
              Play Again
            </button>
          </div>
        </div>
      )}

      {showStartModal && connected && !gameOver && (
        <StartMatchModal onStartGame={handleStartGame} />
      )}

      {showTicketModal && hasPendingTickets && myPlayer?.pendingTickets && (
        <TicketModal
          tickets={myPlayer.pendingTickets}
          onKeep={handleKeepTickets}
        />
      )}

      {gameOver && (
        <GameOverModal
          players={players}
          mode={mode}
          onNewGame={handleReset}
        />
      )}

      {cardChoiceModal && (
        <CardChoiceModal
          route={cardChoiceModal.route}
          options={cardChoiceModal.actions}
          onChoose={(action) => {
            sendAction(action.index);
            setCardChoiceModal(null);
          }}
          onCancel={() => setCardChoiceModal(null)}
        />
      )}

      {/* Main Content: Players Left + Board */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Players Column */}
        {players.length > 0 && (
          <div className="flex flex-col gap-2 p-2 bg-gray-900/50 border-r border-gray-700 overflow-y-auto">
            {players.map((player, idx) => {
              if (!shouldShowPlayer(idx, mode)) return null;

              // Get routes claimed by this player
              const playerRoutes = routes.filter(route => {
                const routeId = getRouteId(route);
                const reverseRouteId = `${route.target}-${route.source}-${route.key}`;
                const claimedBy = claimedRoutes.get(routeId) ?? claimedRoutes.get(reverseRouteId);
                return claimedBy === idx;
              });

              return (
                <div
                  key={idx}
                  onMouseEnter={() => setHoveredPlayer(idx)}
                  onMouseLeave={() => setHoveredPlayer(null)}
                >
                  <PlayerPanel
                    player={player}
                    playerIdx={idx}
                    isCurrentPlayer={currentPlayerIdx === idx}
                    claimedRoutes={playerRoutes}
                  />
                </div>
              );
            })}
          </div>
        )}

        {/* Game Board */}
        <div className="flex-1 flex items-center justify-center p-2 min-h-0 overflow-hidden">
          <svg
            viewBox="0 0 800 541"
            className="max-h-full max-w-full"
          >
            <EuropeMap />

            <g id="routes">
              {routes.map((route, idx) => {
                const routeWaypoints = generateWaypointsFromCities(route, cities, routes);
                if (routeWaypoints.length === 0) return null;
                const routeId = getRouteId(route);
                const reverseRouteId = `${route.target}-${route.source}-${route.key}`;
                const claimedBy = claimedRoutes.get(routeId) ?? claimedRoutes.get(reverseRouteId);

                return (
                  <Route
                    key={`${route.source}-${route.target}-${route.color}-${idx}`}
                    route={route}
                    waypoints={routeWaypoints}
                    claimedBy={claimedBy}
                    onClaim={handleRouteClaim}
                    highlightPlayer={hoveredPlayer}
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

      {/* Face Up Cards */}
      {faceUpCards.length > 0 && (
        <div className="flex items-center justify-center gap-3 px-4 py-3 bg-gradient-to-t from-gray-900 to-gray-800 border-t border-gray-700">
          {isMyTurn && (
            <div className="flex items-center gap-2 mr-2">
              <span className="text-green-400 text-sm font-medium animate-pulse">
                {firstPick ? 'Pick 2nd card!' : 'Your Turn!'}
              </span>
              {firstPick && (
                <button
                  onClick={() => setFirstPick(null)}
                  className="text-xs text-gray-400 hover:text-white px-2 py-0.5 bg-gray-700 rounded"
                >
                  Cancel
                </button>
              )}
              {canDrawTickets && !firstPick && (
                <button
                  onClick={handleDrawTickets}
                  className="text-xs text-white font-medium px-3 py-1.5 bg-purple-600 hover:bg-purple-500 rounded transition-colors"
                >
                  Draw Tickets
                </button>
              )}
            </div>
          )}
          <div
            className={`flex items-center gap-1 mr-2 ${(canPickFirst('deck') || canPickSecond('deck')) ? 'cursor-pointer' : ''}`}
            onClick={() => handleCardPick('deck')}
          >
            <div className={`w-10 h-14 rounded-lg bg-gradient-to-br from-amber-700 to-amber-900 border-2 flex items-center justify-center shadow-lg transition-all ${
              canPickFirst('deck') || canPickSecond('deck')
                ? 'border-green-400 hover:scale-105 hover:shadow-green-400/30'
                : firstPick?.type === 'deck'
                ? 'border-yellow-400 scale-105'
                : 'border-amber-600'
            }`}>
              <span className="text-amber-200 text-lg">🎴</span>
            </div>
            <span className="text-gray-500 text-xs ml-1">Deck</span>
          </div>
          <div className="w-px h-12 bg-gray-600" />
          {faceUpCards.map((card, idx) => {
            const isFirstPick = firstPick?.type === 'face_up' && firstPick?.card === card;
            const clickable = canPickFirst('face_up', card) || canPickSecond('face_up', card);
            return (
              <FaceUpCard
                key={idx}
                color={card}
                clickable={clickable}
                selected={isFirstPick}
                onClick={() => handleCardPick('face_up', card)}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
