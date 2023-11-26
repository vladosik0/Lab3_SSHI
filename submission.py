from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
from typing import Any, DefaultDict, List, Set, Tuple

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState: GameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions(agentIndex):
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState: GameState, action: str) -> float:
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  def getAction(self, gameState: GameState) -> str:
    maxLevel, minLevel = 0, 0
    value, action = self.maxValue(gameState, maxLevel, minLevel)
    return action

  def maxValue(self, state: GameState, maxLevel, minLevel):
    if self.isEnd(state) or maxLevel == self.depth:
      return self.eval(state)
    v = float('-inf'), None
    for action in self.actions(state, self.index):
      tmp_v, tmp_a = v
      successorGameState = state.generateSuccessor(self.index, action)
      agent = 0
      value = self.minValue(successorGameState, maxLevel + 1, minLevel, agent + 1)
      if value > tmp_v:
        v = value, action
    return v

  def minValue(self, state: GameState, maxLevel, minLevel, agent):
    numberOfAgents = state.getNumAgents()
    v = float('inf')
    if minLevel == (numberOfAgents - 1) * self.depth or self.isEnd(state):
      return self.eval(state)
    for action in self.actions(state, agent):
      successorGameState = state.generateSuccessor(agent, action)
      if (numberOfAgents - 1) != agent:
        value = self.minValue(successorGameState, maxLevel, minLevel + 1, agent + 1)
      else:
        value = self.maxValue(successorGameState, maxLevel, minLevel)
      if type(value) is tuple:
        tmp_v, tmp_a = value
      else:
        tmp_v = value
      v = min(v, tmp_v)
    return v

  def eval(self, state: GameState):
    return self.evaluationFunction(state)

  def isEnd(self, state: GameState):
    return state.isWin() or state.isLose()

  def actions(self, state: GameState, agent_index):
    return state.getLegalActions(agent_index)

  # def getLegalActionsNoStop(index, gameState):
  #   possibleActions = gameState.getLegalActions(index)
  #   if Directions.STOP in possibleActions:
  #     possibleActions.remove(Directions.STOP)
  #   return possibleActions


######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    You may reference the pseudocode for Alpha-Beta pruning here:
    en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning#Pseudocode
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState: GameState) -> str:
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState: GameState) -> float:
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  """Calculate distance to the nearest food"""
  newFoodList = np.array(newFood.asList())
  distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
  min_food_distance = 0
  if len(newFoodList) > 0:
    min_food_distance = distanceToFood[np.argmin(distanceToFood)]

  """Calculate the distance to nearest ghost"""
  ghostPositions = np.array(currentGameState.getGhostPositions())
  distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
  min_ghost_distance = 0
  nearestGhostScaredTime = 0
  if len(ghostPositions) > 0:
    min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
    nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
    # avoid certain death
    if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
      return float("-inf")
    # eat a scared ghost
    if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
      return float("inf")

  value = currentGameState.getScore() - min_food_distance
  if nearestGhostScaredTime > 0:
    # follow ghosts if scared
    value -= min_ghost_distance
  else:
    value += min_ghost_distance
  return value


# Abbreviation
better = betterEvaluationFunction
