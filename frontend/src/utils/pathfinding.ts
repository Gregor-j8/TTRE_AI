import type { Route } from '../types/board';

export function areCitiesConnected(
  source: string,
  target: string,
  claimedRoutes: Route[]
): boolean {
  if (source === target) return true;

  const graph = new Map<string, Set<string>>();

  for (const route of claimedRoutes) {
    if (!graph.has(route.source)) {
      graph.set(route.source, new Set());
    }
    if (!graph.has(route.target)) {
      graph.set(route.target, new Set());
    }
    graph.get(route.source)!.add(route.target);
    graph.get(route.target)!.add(route.source);
  }

  const queue: string[] = [source];
  const visited = new Set<string>([source]);

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (current === target) {
      return true;
    }

    const neighbors = graph.get(current);
    if (neighbors) {
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
  }

  return false;
}
