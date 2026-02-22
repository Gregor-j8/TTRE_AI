import math
import copy
import torch
import torch.nn.functional as F


class MCTSNode:
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def q_value(self):
        """Average value of this node (W / N)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct):
        """
        UCB score for selection.

        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        - Q: exploitation (how good has this node been?)
        - The rest: exploration (try less-visited nodes with high prior)
        """
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + exploration

    def is_expanded(self):
        """Has this node been expanded (children created)?"""
        return len(self.children) > 0

    def select_child(self, c_puct):
        """Select the child with highest UCB score"""
        best_score = float('-inf')
        best_child = None
        for child in self.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child


class MCTS:
    def __init__(self, model, encoder, action_to_idx=None, c_puct=1.5, num_simulations=100):
        self.model = model
        self.encoder = encoder
        self.action_to_idx = action_to_idx or {}
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = next(model.parameters()).device

    def _get_action_key(self, action):
        """Convert action to hashable key for action_to_idx lookup"""
        return (action.type, action.source1, action.source2, action.card1, action.card2)

    def run(self, game, player_idx):
        """
        Run MCTS from the current game state.
        Returns: (best_action, policy)
            - best_action: the action with most visits
            - policy: visit count distribution (for training)
        """
        root = MCTSNode()

        # Expand root node first
        self._expand(root, game, player_idx)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_game = copy.deepcopy(game)
            sim_player_idx = player_idx

            # SELECT: Walk down to a leaf
            while node.is_expanded() and not sim_game.game_over:
                node = node.select_child(self.c_puct)
                if node.action:
                    # Check if action is still legal (game state may have diverged)
                    legal_actions = sim_game.get_legal_actions()
                    action_key = self._get_action_key(node.action)

                    # Find matching action in current legal actions
                    valid_action = None
                    for la in legal_actions:
                        if self._get_action_key(la) == action_key:
                            valid_action = la
                            break

                    if valid_action:
                        sim_game.step(valid_action)
                        sim_player_idx = sim_game.current_player_idx
                    else:
                        # Action no longer valid - treat this node as a leaf
                        break

            # EXPAND & EVALUATE
            if not sim_game.game_over:
                value = self._expand(node, sim_game, sim_player_idx)
            else:
                # Game ended - get actual result
                scores = [p.points for p in sim_game.state.list_of_players]
                if scores[player_idx] > scores[1 - player_idx]:
                    value = 1.0
                elif scores[player_idx] < scores[1 - player_idx]:
                    value = -1.0
                else:
                    value = 0.0

            # BACKPROPAGATE
            self._backpropagate(node, value, player_idx, sim_player_idx)

        # Select best action (most visits)
        best_child = max(root.children.values(), key=lambda c: c.visit_count)

        # Create policy from visit counts (for training)
        # Return as list of (action, prob) tuples since Action isn't hashable
        total_visits = sum(c.visit_count for c in root.children.values())
        policy = [
            (child.action, child.visit_count / total_visits)
            for child in root.children.values()
        ]

        return best_child.action, policy

    def _expand(self, node, game, player_idx):
        """
        Expand a node: create children for all legal actions.
        Returns the value estimate from the neural network.
        """
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return 0.0

        # Get neural network evaluation
        data = self.encoder.encode_state(game.state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(data)

        # Convert logits to probabilities
        probs = F.softmax(policy_logits, dim=1).squeeze(0)

        # Get priors from network for each legal action
        action_priors = []
        for action in legal_actions:
            key = self._get_action_key(action)
            if key in self.action_to_idx:
                idx = self.action_to_idx[key]
                if idx < probs.shape[0]:
                    action_priors.append(probs[idx].item())
                else:
                    action_priors.append(1e-6)
            else:
                # Unknown action - give small prior
                action_priors.append(1e-6)

        # Normalize priors to sum to 1
        total = sum(action_priors)
        if total > 0:
            action_priors = [p / total for p in action_priors]
        else:
            # Fallback to uniform
            action_priors = [1.0 / len(legal_actions)] * len(legal_actions)

        # Create child nodes (use action key as dict key since Action isn't hashable)
        for action, prior in zip(legal_actions, action_priors):
            action_key = self._get_action_key(action)
            node.children[action_key] = MCTSNode(
                parent=node,
                action=action,
                prior=prior
            )

        return value.item()

    def _backpropagate(self, node, value, root_player_idx, leaf_player_idx):
        """
        Backpropagate the value up the tree.

        Note: Value needs to be flipped for opponent's nodes!
        In a 2-player game, if value is good for root player,
        it's bad for opponent. We flip the sign as we go up.
        """
        # Value is from perspective of root_player_idx
        # As we walk up, we flip it each level (alternating players)
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            # Flip value for parent (opponent's perspective)
            current_value = -current_value
            node = node.parent
