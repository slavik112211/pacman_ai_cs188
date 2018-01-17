# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

import pdb
import sys
sys.path.insert(0,'..')

"""
1. ReflexAgent:
  tests: python autograder.py -q q1 --no-graphics
         python autograder.py -t test_cases/q2/0-small-tree
  random ghosts: python pacman.py --frameTime 0 -p ReflexAgent -k 2 -l mediumClassic
  out-to-get-you-ghosts: -g DirectionalGhost
  maps: -l openClassic, testClassic, mediumClassic
  
  python pacman.py --frameTime 0 -p ExpectimaxAgent -g DirectionalGhost -l smallClassic
"""
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        gameState.currentClosestFood = getClosestObject(gameState.getPacmanPosition(), gameState.getFood().asList())
        ghostsPositions = [ghostState.getPosition() for ghostState in gameState.getGhostStates()]
        gameState.currentClosestGhost = getClosestObject(gameState.getPacmanPosition(), ghostsPositions)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        # Add some randomness to the process of move selection, to avoid getting stuck behind the walls,
        # when distance to the closest food pellet is not changing
        if(coinToss(0.3)):
            chosenIndex = random.randrange(0,len(scores))
            scores[chosenIndex] += 1

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        # import pdb; pdb.set_trace()
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        additionalScore = 0
        # 1. Does this new state gets us closer to the closest dot? +1 to stateScore if yes.
        currentPos = currentGameState.getPacmanPosition()
        currentDistanceToClosestFood = manhattanDistance(currentPos, currentGameState.currentClosestFood)
        newDistanceToClosestFood = manhattanDistance(newPos, currentGameState.currentClosestFood)
        if(newDistanceToClosestFood < currentDistanceToClosestFood):
            additionalScore += 1

        # 2. If closest ghost distance is <= 3, decrease stateScore
        if(manhattanDistance(newPos, currentGameState.currentClosestGhost) <= 3):
            additionalScore -= 2

        if(manhattanDistance(newPos, currentGameState.currentClosestGhost) <= 1):
            additionalScore -= 1

        # Score calculation: -1 for the move, +10 for eating food, -500 if would get eaten by ghost
        return successorGameState.getScore() + additionalScore


def coinToss(p=.5):
    return True if random.random() < p else False

def getClosestObject(position, objects):
    """
      Get closest object, be it a food pellet, or a ghost
    """
    minDistance = 999999; minIndex = -1
    if len(objects) == 0: return 0
    for index, object in enumerate(objects):
        distance = manhattanDistance(position, object)
        if(distance < minDistance):
            minDistance = distance
            minIndex = index

    return objects[minIndex]

def scoreEvaluationFunction(currentGameState):
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

"""
Q2.
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
-l openClassic mediumClassic
python autograder.py -q q2

"On larger boards such as openClassic and mediumClassic (the default), 
you'll find Pacman to be good at not dying, but quite bad at winning. 
He'll often thrash around without making progress. 
Don't worry if you see this behavior, question 5 will clean up all of these issues."
The reason for this behavior is that evaluation function doesn't take into account 
which move will get it closer to the closest food.
"""
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, gameState, ply, agentIndex):
        if(self.terminalState(gameState, ply)):
            return self.evaluationFunction(gameState)

        value = -999999
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.minValue(successorState, ply, agentIndex+1))

        return value

    def minValue(self, gameState, ply, agentIndex):
        # pdb.set_trace()
        if(self.terminalState(gameState, ply)):
            return self.evaluationFunction(gameState)

        value = 999999
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex+1 == gameState.getNumAgents():  # {0: "pacman", 1: "ghost1", 2: "ghost2"}, getNumAgents=3
                value_new = self.maxValue(successorState, ply+1, 0)
            else: value_new = self.minValue(successorState, ply, agentIndex+1)
            value = min(value, value_new)

        return value

    def terminalState(self, gameState, ply):
        return (not gameState.getLegalActions()) or (ply > self.depth)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # pdb.set_trace()
        legalMoves = gameState.getLegalActions()
        ply = 1; agentIndex = 0;
        scores = [self.minValue(gameState.generateSuccessor(agentIndex, action), ply, agentIndex+1) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
      python autograder.py -q q3
      Compare performance of the two:
      python pacman.py -p AlphaBetaAgent -a depth=4 -l smallClassic (faster, alpha-beta pruning)
      python pacman.py -p MinimaxAgent -a depth=4 -l smallClassic (slower, no pruning)
    """
    def maxValue(self, gameState, ply, agentIndex, alpha, beta):
        if(self.terminalState(gameState, ply)):
            return {'value': self.evaluationFunction(gameState), 'action': ''}


        bestMax = {'value': -float("inf"), 'action': ''}
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            newMove = self.minValue(successorState, ply, agentIndex+1, alpha, beta)
            if(newMove['value'] > bestMax['value']):
                bestMax['value'] = newMove['value']
                bestMax['action'] = action
            if(bestMax['value'] > beta): return bestMax
            alpha = max(alpha, bestMax['value'])

        return bestMax

    def minValue(self, gameState, ply, agentIndex, alpha, beta):
        if(self.terminalState(gameState, ply)):
            return {'value': self.evaluationFunction(gameState), 'action': ''}

        bestMin = {'value': float("inf"), 'action': ''}
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex+1 == gameState.getNumAgents():  # {0: "pacman", 1: "ghost1", 2: "ghost2"}, getNumAgents=3
                newMove = self.maxValue(successorState, ply+1, 0, alpha, beta)
            else: newMove = self.minValue(successorState, ply, agentIndex+1, alpha, beta)
            if(newMove['value'] < bestMin['value']):
                bestMin['value'] = newMove['value']
                bestMin['action'] = action
            if(bestMin['value'] < alpha): return bestMin
            beta = min(beta, bestMin['value'])

        return bestMin

    def terminalState(self, gameState, ply):
        return (not gameState.getLegalActions()) or (ply > self.depth)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        ply = 1; agentIndex = 0;
        bestMove = self.maxValue(gameState, ply, agentIndex, -float("inf"), float("inf"))
        return bestMove['action']

# Pacman fun: python pacman.py --frameTime 0 -p ExpectimaxAgent -g DirectionalGhost -l smallClassic
# make sure it's using the evalFn = 'better', and search depth='2' or '3'
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
      python autograder.py -q q4
    """
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        # evalFn = 'better'
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


    def maxValue(self, gameState, ply, agentIndex, alpha, beta):
        if(self.terminalState(gameState, ply)):
            return {'value': self.evaluationFunction(gameState), 'action': ''}

        scores = list()
        legalActions = gameState.getLegalActions(agentIndex)
        successorStates = list()
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorStates.append(successorState)
            newMove = self.expectimaxValue(successorState, ply, agentIndex+1, alpha, beta)
            scores.append(newMove['value'])

        bestScore = max(scores)
        # if(scores == [333.0, 343.0, 333.0, 343.0]): pdb.set_trace()
        # if(ply == 1): print "scores: " + str(scores) + "; actions: "+ str(legalActions)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return {'value': bestScore, 'action': legalActions[chosenIndex]}

    def expectimaxValue(self, gameState, ply, agentIndex, alpha, beta):
        if(self.terminalState(gameState, ply)):
            return {'value': self.evaluationFunction(gameState), 'action': ''}

        scores = list()
        legalActions = gameState.getLegalActions(agentIndex)
        for index, action in enumerate(legalActions):
            successorState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex+1 == gameState.getNumAgents():  # {0: "pacman", 1: "ghost1", 2: "ghost2"}, getNumAgents=3
                newMove = self.maxValue(successorState, ply+1, 0, alpha, beta)
            else: newMove = self.expectimaxValue(successorState, ply, agentIndex+1, alpha, beta)
            scores.append(newMove['value'])

            # if(bestMin['value'] < alpha): return bestMin
            # beta = min(beta, bestMin['value'])

        return {'value': sum(scores)/len(scores), 'action': ''}

    def terminalState(self, gameState, ply):
        return (not gameState.getLegalActions()) or (ply > self.depth)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        ply = 1; agentIndex = 0;
        bestMove = self.maxValue(gameState, ply, agentIndex, -float("inf"), float("inf"))
        return bestMove['action']


# return value in [0,1] range
def normalizeByRange(max, min, current):
    if current < min: current = min
    if current > max: current = max
    if max == min: return max
    return float(current - min) / float(max - min)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    closestFoodPos = getClosestObject(pacmanPos, foodPos) if len(foodPos) else 0
    closestFoodDistance = manhattanDistance(pacmanPos, closestFoodPos) if len(foodPos) else 0

    # Feature 1. Distance to closest food
    # closestFoodScore = 1 - normalizeByRange(10, 0, float(closestFoodDistance)) #Range: [0, 1]; 1=best; 0=worst.
    closestFoodScore = closestFoodDistance


    # Feature 2. Number of food pellets
    # foodLeftScore = 1 - normalizeByRange(100, 0, float(len(foodPos)))
    foodLeftScore = len(foodPos)

    # ghostsPositions = [ghostState.getPosition() for ghostState in gameState.getGhostStates()]
    # gameState.currentClosestGhost = getClosestObject(gameState.getPacmanPosition(), ghostsPositions)
    # Score calculation: -1 for the move, +10 for eating food, -500 if would get eaten by ghost


    weight1 = 1; weight2 = -1; weight3 = -1
    # print "score: "+str(weight1*currentGameState.getScore())+\
    #       "; closest food: "+str(weight2*closestFoodScore)+\
    #       "; food left: "+str(weight3*foodLeftScore)+\
    #       "; total: " + str(weight1*currentGameState.getScore() + weight2*closestFoodScore + weight3 * foodLeftScore)#

    return weight1 * currentGameState.getScore() + weight2 * closestFoodScore + weight3 * foodLeftScore


# Abbreviation
better = betterEvaluationFunction
