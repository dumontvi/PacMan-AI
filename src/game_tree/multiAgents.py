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
        nextState = (gameState.generatePacmanSuccessor(legalMoves[chosenIndex]))

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
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        currentFood = currentGameState.getFood()
        currentCapsules = currentGameState.getCapsules()

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newGhostPos = successorGameState.getGhostPositions()
        foodPos = newFood.asList()
        currentFoodPos = currentFood.asList()

        if successorGameState.isWin():
            return 1e5

        if newScaredTimes[0] > 0:
            ghostDist = 0
        else:
            ghostDist = min([manhattanDistance(ghost, newPos) for ghost in newGhostPos])
            if ghostDist < 2:
                ghostDist = -1e10
            else:
                ghostDist = 0

        # Food
        if len(foodPos) == 0 or len(foodPos) < len(currentFoodPos):
          foodDist = 1e-5
        else:  
          foodDist = min([manhattanDistance(food, newPos) for food in foodPos])
        

        return ghostDist + 1/float(foodDist)

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
        action, score = self.minimax(gameState)

        return action

    def minimax(self, gameState, treeDepth = 0, agent = 0):        
        best_move = None 
        totalDepth = self.depth * gameState.getNumAgents() # 16, where 16 is pacman
        if agent >= gameState.getNumAgents():
            agent = agent % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose() or treeDepth >= totalDepth: 
            return best_move, self.evaluationFunction(gameState)

        if agent == 0: value = float("-inf")
        if agent > 0: value = float("inf")

        for move in gameState.getLegalActions(agent):
            next_gameState = gameState.generateSuccessor(agent, move)
            
            next_move, next_value = self.minimax(next_gameState, treeDepth + 1, agent + 1)

            if agent == 0 and value < next_value:
                value, best_move = next_value, move
            if agent > 0 and value > next_value:
                value, best_move = next_value, move

        return best_move, value 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.minimax(gameState)
        return action

    def minimax(self, gameState, treeDepth = 0, agent = 0, alpha=float("-inf"), beta=float("inf")):        
        best_move = None 
        totalDepth = self.depth * gameState.getNumAgents() # 16, where 16 is pacman
        if agent >= gameState.getNumAgents():
            agent = agent % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose() or treeDepth >= totalDepth: 
            return best_move, self.evaluationFunction(gameState)

        if agent == 0: value = float("-inf")
        if agent > 0: value = float("inf")

        for move in gameState.getLegalActions(agent):
            next_gameState = gameState.generateSuccessor(agent, move)
            
            next_move, next_value = self.minimax(next_gameState, treeDepth + 1, agent + 1, alpha, beta)

            if agent == 0: 
                if value < next_value:
                    value, best_move = next_value, move
                if value >= beta: return best_move, value 
                alpha = max(alpha, value)   
            if agent > 0:
                if value > next_value:
                    value, best_move = next_value, move
                if value <= alpha: return best_move, value
                beta = min(value, beta)    

        return best_move, value  

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
        action, score = self.expectimax(gameState)
        return action

    def expectimax(self, gameState, treeDepth = 0, agent = 0):
        best_move = None 
        totalDepth = self.depth * gameState.getNumAgents() # 16, where 16 is pacman
        if agent >= gameState.getNumAgents():
            agent = agent % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose() or treeDepth >= totalDepth: 
            return best_move, self.evaluationFunction(gameState)

        if agent == 0: value = float("-inf")
        if agent > 0: value = float(0)
        
        prob = float(1.0)/float(len(gameState.getLegalActions(agent)))
        for move in gameState.getLegalActions(agent):
            next_gameState = gameState.generateSuccessor(agent, move)
            
            next_move, next_value = self.expectimax(next_gameState, treeDepth + 1, agent + 1)
            if agent == 0 and value < next_value:
                value, best_move = next_value, move
            if agent > 0:
                value = value + prob * next_value

        return best_move, value       

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Evaluation Function: 10 * closestFood + 
                           ghostPosition + 
                           100 * numFood + 
                           100 * numCapsules + 
                           currentGameState.getScore()

      DESCRIPTION: 
      The evaluation function was uses the following 5 features:
      - Distance from closest food
      - Distance from closest ghost
      - Number of Food 
      - Number of Capsules
      - Current Score

      Other features that were tested but didn't add value:
      - Distance from Capsules
      - Distance from Walls
      - Other interactional features like multiplication of above features

      The evaulation function is the weighted linear combination of the stated five features.
      We use the reciprocal for features like the closest food, number of food and 
      capsules to maximize the evaluation function. I also assigned weights to the features closestFood, 
      numFood and numCapsules, which helped increase my score. I kept averaging below 1000, but increasing
      the weight of the numCapsules greatly boosted my score. Adding this filter "newScaredTimes[0] == 0"
      when checking for nearby ghosts also increased my score.
    """
    "*** YOUR CODE HERE ***"
    
    foodList = currentGameState.getFood().asList()
    currentCapsules = currentGameState.getCapsules()
    pacmanPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Food Dist from Pacman
    if len(foodList) == 0:
        closestFood = 1e-5
    else:  
        closestFood = (1/float(min([manhattanDistance(food, pacmanPos) for food in foodList])) + 1/float(random.randint(10, 12)))
    
    # Ghost Dist from Pacman
    ghostPos = currentGameState.getGhostPositions()
    ghostPosition = min([manhattanDistance(ghost, pacmanPos) for ghost in ghostPos])
    if ghostPosition < 2 and newScaredTimes[0] == 0: # ghost is too close and isn't scared
        ghostPosition = -1e10
    else:
        ghostPosition = 0 

    # Number of Food
    numFood = 1/float(len(foodList) + 1) # +1 to prevent 0 float division

    # Number of Capsule
    numCapsules = 1/float(len(currentCapsules) + 1) # +1 to prevent 0 float division

    # Tested with different weights. Averaged above 1000 after I assigned high weight value to numCapsules
    foodGhostCapsulescore = 10 * closestFood + ghostPosition + 100 * numFood + 100 * numCapsules
    additionalScoring = foodGhostCapsulescore  + currentGameState.getScore()

    return additionalScoring

# Abbreviation
better = betterEvaluationFunction

