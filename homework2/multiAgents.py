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


from autograder import printTest
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        capsuleList = successorGameState.getCapsules()
        score = successorGameState.getScore()

        # Penalty for stopping
        if action == 'Stop':
            score -= 500

        # Food distance 
        foodList = newFood.asList()
        if bool(foodList):
            newDistanceToFoods = [manhattanDistance(newPos, food) for food in foodList]
            maxDistanceFood = max(newDistanceToFoods)
            score -= maxDistanceFood
        
        # Capsule distance
        if bool(capsuleList) :
            newDistanceToCapsules = [manhattanDistance(newPos, capsule) for capsule in capsuleList]
            maxCapsuleDistance = max(newDistanceToCapsules) 
            score -= maxCapsuleDistance
        
        # Ghost distance
        distanceToGhosts = []
        distanceToScaredGhosts = []

        for ghost in newGhostStates:
            if  ghost.scaredTimer == 0: # not scared ghosts 
                distanceToGhosts.append(manhattanDistance(newPos, ghost.getPosition())) 
            else:                       # scared ghosts 
                score += 100     # eat capsule                 
                distanceToScaredGhosts.append(manhattanDistance(newPos, ghost.getPosition())) 
        
        if bool(distanceToGhosts):
            closestGhostDistance = min(distanceToGhosts)
            
            # Don't die !
            if closestGhostDistance == 0:
                score -= 1000
            
            score += closestGhostDistance
        
        # Eat or run away from ghost
        if bool(distanceToScaredGhosts):
            for ghostIndex in range(len(distanceToScaredGhosts)):
                if distanceToScaredGhosts[ghostIndex] < newScaredTimes[ghostIndex]:
                    score -= distanceToScaredGhosts[ghostIndex]
                else:
                    score += distanceToScaredGhosts[ghostIndex]
         
        return score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
    
        ## Helper Functions ## 
        
        # Check the state is leaf state
        def isTerminal(state, depth):
            return self.depth == depth or state.isWin() or state.isLose()

        # MAX-VALUE function
        def maxValue(state, depth, agentIndex):
            
            if isTerminal(state, depth):
                return self.evaluationFunction(state)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()

            v = -float("inf")
            actions = state.getLegalActions(agentIndex)
            
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successor, depth, nextAgentIndex))
            return v

        # MIN-VALUE function
        def minValue(state, depth, agentIndex):
            
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            v = float("inf")
            actions = state.getLegalActions(agentIndex)
            
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                
                # Next layer is a Max layer
                if nextAgentIndex == 0:
                    v = min(v, maxValue(successor, depth + 1, nextAgentIndex))
                
                # Next layer is also a Min layer
                else:
                    v = min(v, minValue(successor, depth, nextAgentIndex))

            return v    
        
        ## MINIMAX-DECISION ## 

        pacman = 0
        pacmanActions = gameState.getLegalActions(pacman)
        depth = 0
        
        maxAction = -float("inf")
        for action in pacmanActions:
            successor = gameState.generateSuccessor(pacman, action)
            value = minValue(successor, depth, pacman + 1)
                        
            # argmax action
            if value > maxAction:
                maxAction = value
                decision = action
            
        
        return decision

        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        ## Helper Functions ## 
        
        # Check the state is leaf state
        def isTerminal(state, depth):
            return self.depth == depth or state.isWin() or state.isLose()

        # MAX-VALUE function
        def maxValue(state, depth, agentIndex, alpha, beta):
            
            if isTerminal(state, depth):
                return self.evaluationFunction(state)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()

            v = -float("inf")
            actions = state.getLegalActions(agentIndex)
            
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successor, depth, nextAgentIndex, alpha, beta))

                # Alpha - Beta Pruning
                if v > beta:
                    return v

                alpha = max(v, alpha)

            return v

        # MIN-VALUE function
        def minValue(state, depth, agentIndex, alpha, beta):
            
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
            v = float("inf")
            actions = state.getLegalActions(agentIndex)
            
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                
                # Next layer is a Max layer
                if nextAgentIndex == 0:
                    v = min(v, maxValue(successor, depth + 1, nextAgentIndex, alpha, beta))
                
                # Next layer is also a Min layer
                else:
                    v = min(v, minValue(successor, depth, nextAgentIndex, alpha, beta))
                
                # Alpha - Beta Pruning
                if v < alpha:
                    return v
                beta = min(beta, v)

            return v    
        
        ## ALPHA-BETA-SEARCH ## 

        pacman = 0
        pacmanActions = gameState.getLegalActions(pacman)
        depth = 0

        # Alpha - Beta values
        alpha = -float("inf") # best value for max agent
        beta = float("inf") # best value for min agent 
        
        maxAction = -float("inf")
        for action in pacmanActions:
            successor = gameState.generateSuccessor(pacman, action)
            value = minValue(successor, depth, pacman + 1, alpha, beta)
                        
            # argmax action
            if value > maxAction:
                maxAction = value
                decision = action
                alpha = value
            
        
        return decision


    

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

        ## Helper Functions ## 
        
        # Check the state is leaf state
        def isTerminal(state, depth):
            return self.depth == depth or state.isWin() or state.isLose()

        # MAX-VALUE function
        def maxValue(state, depth, agentIndex):
            
            if isTerminal(state, depth):
                return self.evaluationFunction(state)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()

            v = -float("inf")
            actions = state.getLegalActions(agentIndex)

            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successor, depth, nextAgentIndex))

            return v

        # MIN-VALUE function
        def minValue(state, depth, agentIndex):
            
            if isTerminal(state,depth):
                return self.evaluationFunction(state)
            
            nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
           
            actions = state.getLegalActions(agentIndex)

            # Exceptation
            v = 0
            p = 1 / len(actions) # uniform
            
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                
                # Next layer is a Max layer
                if nextAgentIndex == 0:
                    v += maxValue(successor, depth + 1, nextAgentIndex) * p
                
                # Next layer is also a Min layer
                else:
                    v += minValue(successor, depth, nextAgentIndex) * p
                
            return v    
        
        ## EXPECTIMINIMAX-SEARCH ## 

        pacman = 0
        pacmanActions = gameState.getLegalActions(pacman)
        depth = 0

        maxAction = -float("inf")
        for action in pacmanActions:
            successor = gameState.generateSuccessor(pacman, action)
            value = minValue(successor, depth, pacman + 1)
                        
            # argmax action
            if value > maxAction:
                maxAction = value
                decision = action            
        
        return decision

        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 

    EVAL = sum [ w_i * f_i ] for i = 1:5.
 
    -f1 = SCORE from the game itself: time + win/lose + (eat food and ghost)
    -f2 = 1 / MIN FOOD DISTANCE
    -f3 = 1 / MIN CAPSULE DISTANCE
    -f4 = MIN INEDIBLE-GHOST DISTANCE
    -f5 = MIN EDIBLE-GHOST

    -w1 = 1
    -w2 = 4
    -w3 = 5
    -w4 = 0.2
    -w5 = 8
    
    Linear combination method for evalation score is a felixable and good approxation
    function and suitable method for pacman game. 
    
    For the features; the score of the game, closest food distance, 
    closest capsule distance, closest edible and inedible ghost distance 
    are used. f1 gives a good average score for the game state and takes the time and 
    win/lose status of the game in to calculations. Altought it captures eating a food and 
    ghost points, its not enough for good evalation, therefore the distance to closest food 
    solves the case. The agent will want to decrease the min distance since its inverse 
    proportionally added to score therefore it will eat the food. Capsules have same 
    approach with food, additionally the weight is larger due to the importance of a capsule.
    Distance to inedible ghost has a small weight since the agent can come close to ghost unless 
    it won't reach to agent. In addition if the closest distance to the ghosts is 
    equal to zero which means same location feature score will 1000 for penalty purposes. 
    Edible ghost has same approch with food however their weight has higher absolute value 
    since eating a ghost is is better than eating food if the ghost is close. 

    For the weights, the importance of food and distance has thought, altought the precision 
    of the weights are experimental.

    """

    "*** YOUR CODE HERE ***"
    # Linear Combination variables
    features = []
    weights = []

    # Get game data
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    
    ## SCORE FROM GAME ##
    # f1, w1
    features.append(currentGameState.getScore())
    weights.append(1)
    
    ## FOOD ## 
    foodList = newFood.asList()
   
    # Food distance
    if bool(foodList):
        distanceToFoods = [manhattanDistance(newPos, food) for food in foodList]
        minDistanceFood = min(distanceToFoods) 
        
        #f2, w2
        features.append(1 / minDistanceFood)
        weights.append(4)
    else:
        # Win the game
        features.append(400)
        weights.append(1)

    
    ## CAPSULE ## 
    capsuleList = currentGameState.getCapsules()

    # Capsule distance
    if bool(capsuleList) :
        distanceToCapsules = [manhattanDistance(newPos, capsule) for capsule in capsuleList]
        minCapsuleDistance = min(distanceToCapsules) 
        
        # f3, w3
        features.append(1 / minCapsuleDistance)
        weights.append(5)
    

    ## GHOST DISTANCE ## 
    distanceToGhosts = []
    distanceToScaredGhosts = [] 
    
    for ghost in newGhostStates:
        # not scared and inedible  ghosts
        isFar = ghost.scaredTimer < manhattanDistance(newPos, ghost.getPosition())
        if  ghost.scaredTimer == 0 or isFar: 
            distanceToGhosts.append(manhattanDistance(newPos, ghost.getPosition()))     
        # scared and edible  ghosts
        else:         
            distanceToScaredGhosts.append(manhattanDistance(newPos, ghost.getPosition())) 
       
    # Run away from ghost
    if bool(distanceToGhosts):
        minGhostDistance = min(distanceToGhosts)
        
        # f4, w4
        if minGhostDistance == 0: # Don't die !
            features.append(1000)
            weights.append(-1)
        else:
            features.append(minGhostDistance)
            weights.append(0.2)
    
    # Eat ghost
    if bool(distanceToScaredGhosts):
            minScaredDistance = min(distanceToScaredGhosts)
            
            # f5, w5
            features.append(1 / minScaredDistance)
            weights.append(8)
    
    # Dot product 
    score = sum([fi*wi for (fi, wi) in zip(features, weights)])
    return score


# Abbreviation
better = betterEvaluationFunction
