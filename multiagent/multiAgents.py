# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

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

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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
        "*** YOUR CODE HERE ***"
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        current_food = currentGameState.getFood()
        curFoodList = current_food.asList()
        current_positon = currentGameState.getPacmanPosition()
        new_position = successorGameState.getPacmanPosition()
        new_food = successorGameState.getFood()
        new_ghost_states = successorGameState.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]
        new_food_list = new_food.asList()

        ghostPositions = successorGameState.getGhostPositions()
        distance = float("inf")
        scared = new_scared_times[0] > 0
        
        for ghost in ghostPositions:
          d = manhattanDistance(ghost, new_position)
          distance = min(d, distance)
          #print distance
        
        distance2 = float("inf")        
        distance3 = float("-inf")
        distance4 = float("inf")
        for food in new_food_list:
          d = manhattanDistance(food, new_position)
          d0 = manhattanDistance(food, current_positon)
          distance2 = min(d, distance2)
          distance3 = max(d, distance3)

        #print distance2
        #print distance3
        #print distance4

        condition = len(new_food_list) < len(curFoodList)
        count = len(new_food_list)
        if condition:
          count = 10000
        if distance < 2:
          distance = -100000
        else:
          distance = 0
        if count == 0:
          count = -1000
        if scared:
          distance = 0
        return distance + 1.0/distance2 + count - successorGameState.getScore()
        
        # return successorGameState.getScore()

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        "*** YOUR CODE HERE ***"

        best = float("-inf")
        bestAction = []
        
        legalActions = gameState.getLegalActions(0)

        successors = [(action, gameState.generateSuccessor(0, action)) for action in legalActions]

        for successor in successors:
            val = minimax(1, range(gameState.getNumAgents()), successor[1], self.depth, self.evaluationFunction)
            
            if val > best:
              best = val
              bestAction = successor[0]

        return bestAction
               
def minimax(agent, agentList, state, depth, evaluationFunction):
  
  if depth <= 0 or state.isWin() == True or state.isLose() == True:
    return evaluationFunction(state)
    
  if agent == 0:
    result = float("-inf")
  else:
    result = float("inf")
          
  actions = state.getLegalActions(agent)
  successors = [state.generateSuccessor(agent, action) for action in actions]

  for successor in successors:

    if agent == 0:
      temp = minimax(agentList[agent+1], agentList, successor, depth, evaluationFunction)
      if temp > result:
        result = temp
    else:
      if agent == agentList[-1]:
        temp = minimax(agentList[0], agentList, successor, depth - 1, evaluationFunction)
      else:
        temp = minimax(agentList[agent+1], agentList, successor, depth, evaluationFunction)

      if temp < result:
        result = temp
  
  return result       
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getValue(gameState, 0, float("-inf"), float("inf"), self.depth - 1)[1]
    def getValue(self, gameState, agentIndex, alpha, beta, depth):
        #end on child node
        if gameState.isWin() or gameState.isLose(): # if game should be over
            return (self.evaluationFunction(gameState), Directions.STOP)
        elif agentIndex == gameState.getNumAgents():
            return self.getValue(gameState, 0, alpha, beta, depth - 1)
        elif agentIndex > 0:
            return self.minvalue(gameState, agentIndex, alpha, beta, depth)
        elif agentIndex == 0: #agent is pacman
            return self.maxvalue(gameState, agentIndex, alpha, beta, depth)
        return 0
    def maxvalue(self, gameState, agentIndex, alpha, beta, depth):
        v = float("-inf")
        best_move = Directions.STOP
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            nextMoveScore = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, alpha, beta,
                                          depth)
            if nextMoveScore[0] > v:
                v = nextMoveScore[0]
                best_move = action
                if v > beta:
                    return (v, best_move)
                alpha = max(v, alpha)
        return (v, best_move)
    def minvalue(self, gameState, agentIndex, alpha, beta, depth):
        v = float("inf")
        best_move = Directions.STOP
        if agentIndex == (gameState.getNumAgents() - 1) and depth == 0:
            actions = gameState.getLegalActions(agentIndex) # Collect legal moves and successor states
            for action in actions:
                nextMoveScore = (self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)), action)
                if nextMoveScore[0] < v:
                    best_move = action
                    v = nextMoveScore[0]
                    if v < alpha:
                        return (v, best_move)
                    beta = min(beta, v)
            return (v, best_move)
        else: 
            actions = gameState.getLegalActions(agentIndex) 
            for action in actions:
                nextMoveScore = self.getValue(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, alpha,
                                              beta, depth)
                if nextMoveScore[0] < v:
                    v = nextMoveScore[0]
                    best_move = action
                    if v < alpha:
                        return (v, best_move)
                    beta = min(beta, v)
            return (v, best_move)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        v = float('-inf')
        nextAction = Directions.STOP
        for action in gameState.getLegalActions(0):
                temp = self.expValue(0, 1, gameState.generateSuccessor(0, action))
                if temp > v and action != Directions.STOP:
                        v = temp
                        nextAction = action
       
        return nextAction
               
 
    def maxValue(self, depth, agent, state):
                if depth == self.depth:
                        return self.evaluationFunction(state)
                else:
                        actions = state.getLegalActions(agent)
                        if len(actions) > 0:
                                v = float('-inf')
                        else:
                                v = self.evaluationFunction(state)
                        for action in state.getLegalActions(agent):
                                v = max(v, self.expValue(depth, agent+1, state.generateSuccessor(agent, action)))
                                       
                        return v
 
    def expValue(self, depth, agent, state):
                if depth == self.depth:
                        return self.evaluationFunction(state)
                else:
                        v = 0;
                        actions = state.getLegalActions(agent)
                        for action in actions:
                                if agent == state.getNumAgents() - 1:
                                        v += self.maxValue(depth+1, 0, state.generateSuccessor(agent, action))
                                else:
                                        v += self.expValue(depth, agent+1, state.generateSuccessor(agent, action))
                        if len(actions) != 0:
                                return v / len(actions)
                        else:
                                return self.evaluationFunction(state)        
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
 
    foodDist = 0
    for food in newFood.asList():
      dist = manhattanDistance(food, newPos)
                #print ('dist', dist)
      foodDist += dist
       
    score = 0
    if len(newFood.asList()) == 0:
      score = 1000000000
       
    ghostScore = 0
    if newScaredTimes[0] > 0:
      ghostScore += 100.0
    for state in newGhostStates:
      dist = manhattanDistance(newPos, state.getPosition())
      if state.scaredTimer == 0 and dist < 3:
        ghostScore -= 1.0 / (3.0 - dist);
      elif state.scaredTimer < dist:
        ghostScore += 1.0 / (dist)
       
    score += 1.0 / (1 + len(newFood.asList())) + 1.0 / (1 + foodDist) + ghostScore + currentGameState.getScore()
       
    return score;    
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

