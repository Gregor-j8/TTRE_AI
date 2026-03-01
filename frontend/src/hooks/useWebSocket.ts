import { useEffect, useRef, useCallback, useState } from 'react';
import { useGameState, type PlayerInfo, type LegalAction } from './useGameState';

interface ServerState {
  currentPlayerIdx: number;
  claimedRoutes: Record<string, number>;
  players: PlayerInfo[];
  faceUpCards: string[];
  gameOver: boolean;
  finalRound: boolean;
  legalActions: LegalAction[];
}

interface WebSocketMessage {
  type: string;
  data: unknown;
}

export function useWebSocket(gameId: string = 'default') {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<'visualizer' | 'singleplayer'>('visualizer');
  const { setStateFromServer } = useGameState();

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/game/${gameId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log('WebSocket connected');
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('WebSocket disconnected');
    };

    ws.onmessage = (event) => {
      const message: WebSocketMessage = JSON.parse(event.data);

      if (message.type === 'state') {
        setStateFromServer(message.data as ServerState);
      } else if (message.type === 'game_over') {
        console.log('Game over:', message.data);
      } else if (message.type === 'your_turn') {
        console.log('Your turn!', message.data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, [gameId, setStateFromServer]);

  const startGame = useCallback((gameMode: 'visualizer' | 'singleplayer') => {
    setMode(gameMode);
    wsRef.current?.send(JSON.stringify({
      type: 'start_game',
      mode: gameMode,
    }));
  }, []);

  const sendAction = useCallback((actionIndex: number) => {
    wsRef.current?.send(JSON.stringify({
      type: 'player_action',
      actionIndex,
    }));
  }, []);

  return {
    connected,
    mode,
    startGame,
    sendAction,
  };
}
