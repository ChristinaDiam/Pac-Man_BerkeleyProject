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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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


        from util import manhattanDistance

        score = 0   # Initialize the score with 0

        # Check if the game is won
        if successorGameState.isWin():
            return 999999


        # Calculate Manhattan distance to available food from the successor state
        foodList = newFood.asList()
        foodDistance = [manhattanDistance(newPos, pos) for pos in foodList]

        # Calculate Manhattan distance to each ghost in the game from the successor state
        ghostPos = [ghost.getPosition() for ghost in newGhostStates]
        ghostDistance = [manhattanDistance(newPos, pos) for pos in ghostPos]

        # Calculate Manhattan distance to each ghost in the game from the current state
        ghostPosCurrent = [ghost.getPosition() for ghost in currentGameState.getGhostStates()]
        ghostDistanceCurrent = [manhattanDistance(newPos, pos) for pos in ghostPosCurrent]


        numOfFoodLeft = len(foodList)  # Get the number of food available in the successor state

        numOfFoodLeftCurrent = len(currentGameState.getFood().asList())  # Get the number of food available in the current state

        sumScaredTimes = sum(newScaredTimes)    # Get the total scared time of ghosts in the successor state


        # Calculate the score difference between successor and current states
        score += successorGameState.getScore() - currentGameState.getScore()


        # Give a penalty if the action is to stop
        if action == Directions.STOP:
            score -= 10


        # Add score if there are fewer food items available in the successor state
        if numOfFoodLeft < numOfFoodLeftCurrent:
            score += 200


        # Subtract score for each remaining food item
        score -= 10 * numOfFoodLeft


        # Adjust score based on ghost distances and scared times
        if sumScaredTimes > 0:
            # If ghosts are scared, prefer actions that minimize the distance to ghosts
            score += 200 if min(ghostDistanceCurrent) < min(ghostDistance) else -100
        else:
            # If ghosts are not scared, prefer actions that maximize the distance to ghosts
            score += 200 if min(ghostDistanceCurrent) < min(ghostDistance) else -100

        return score # Returns total score



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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        # The main minimax decision-making function.
        def minimaxDecision(gameState, depth, agentIndex):

            # If terminal state or maximum depth reached, evaluate the state.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Get legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            # If agentIntex is 0, it's Pacman's move (maximizing player)
            if agentIndex == 0:
                maxvalue = float('-inf')        # Initialize max value
           
                for action in legalActions:
                    successor = gameState.generateSuccessor(0,action)
                    maxvalue = max (maxvalue,minimaxDecision(successor, depth, 1))

                return maxvalue
            
            # Ghosts move (minimizing players)
            else:
                minvalue = float('inf')         # Initialize min value

                for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex,action)

                    if agentIndex == gameState.getNumAgents() - 1:
                        # Last ghost, move to the next depth for Pacman.
                        currDepth = depth + 1
                        minvalue = min (minvalue,minimaxDecision(successor,currDepth,0))

                    else:
                        # Move to the next ghost in line.
                        minvalue = min (minvalue,minimaxDecision(successor,depth,agentIndex+1))

                return minvalue

        
        # Get legal actions for Pacman at the root level.
        pacmanActions = gameState.getLegalActions(0)
        currScore = float('-inf')     # Initialize the current score to negative infinity.
        maxScoreAction = ''           # Initialize the variable to an empty string.

        for action in pacmanActions:
            # Generate successor state for Pacman's action.
            nextState = gameState.generateSuccessor(0, action)
            
            # Calculate the score using the minimax decision function.
            score = minimaxDecision(nextState, 0, 1)

            # Choose the action with the maximum score.
            if score > currScore:
                maxScoreAction = action
                currScore = score

        return maxScoreAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Agent intex is 0, it's Pacman's move (maximizer)
        def maxValue(gameState,depth,alpha, beta):

            # If terminal state or maximum depth reached, evaluate the state.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
                
            # Get legal actions
            legalActions = gameState.getLegalActions(0)
            
            v = float('-inf')
            a = alpha
            
            for action in legalActions:
                successor = gameState.generateSuccessor(0,action)
                v = max (v,minValue(successor, depth, 1, a, beta))

                if v > beta:
                    # Prune the search if the current value (v) is greater than beta
                    return v
                
                a = max (a,v)

            return v

        # Ghost's move (minimizer)
        def minValue(gameState,depth,agentIndex,alpha,beta):

            # If terminal state or maximum depth reached, evaluate the state.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Get legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            v = float('inf')
            b = beta

            for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex,action)
                    currDepth = depth + 1

                    if agentIndex == (gameState.getNumAgents() - 1):
                        # Last ghost, move to the next depth for Pacman.
                        v = min (v,maxValue(successor, currDepth, alpha, b))

                        if v < alpha:
                            # Prune the search if the current value (v) is less than alpha
                            return v
                        
                        b = min(b,v)
                    else:
                        # Move to the next ghost in line.
                        v = min (v,minValue(successor,depth,agentIndex+1, alpha, b))

                        if v < alpha:
                            # Prune the search if the current value (v) is less than alpha
                            return v
                        
                        b = min(b,v)
            return v
        

        # Get legal actions for Pacman at the root level.
        pacmanActions = gameState.getLegalActions(0)

        currScore = float('-inf')     # Initialize the current score to negative infinity.
        maxScoreAction = ''           # Initialize the variable to an empty string.
        alpha = float('-inf')   # Initialize alpha to negative infinity.
        beta = float('inf')     # Initialize beta to infinity.

        for action in pacmanActions:
            # Generate successor state for Pacman's action.
            nextState = gameState.generateSuccessor(0,action)
            
            # Next level is a min level. (calling min for successors of the root)
            score = minValue(nextState,0,1,alpha,beta)

            # Choose the action with the maximum score.
            if score > currScore:
                maxScoreAction = action
                currScore = score

            # Updating alpha value at root.    
            if score > beta:
                return maxScoreAction
            alpha = max(alpha,score)

        return maxScoreAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Agent intex is 0, it's Pacman's move (maximizer)
        def maxValue(gameState,depth):

            # If terminal state or maximum depth reached, evaluate the state.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
                
            # Get legal actions and number of legal actions
            legalActions = gameState.getLegalActions(0)
            maxv = float('-inf')
            
            for action in legalActions:
                successor = gameState.generateSuccessor(0,action)
                maxv = max (maxv,expectValue(successor,depth,1))

            return maxv

        # Ghost's move (minimizer)
        def expectValue(gameState,depth,agentIndex):

            # If terminal state or maximum depth reached, evaluate the state.
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Get legal actions
            legalActions = gameState.getLegalActions(agentIndex)
            numoflegalActions = len(legalActions)

            expectval = float('inf')    # Initialize expected value to infinity
            sumOfexpectvalues = 0       # Variable to keep the sum of expected values

            for action in legalActions:
                    successor = gameState.generateSuccessor(agentIndex,action)
                    currDepth = depth + 1

                    if agentIndex == (gameState.getNumAgents() - 1):
                        # Last ghost, move to the next depth for Pacman.
                        expectval = maxValue(successor,currDepth)

                    else:
                        # Move to the next ghost in line.
                        expectval = expectValue(successor,depth,agentIndex+1)

                    sumOfexpectvalues = sumOfexpectvalues + expectval   # Add to the sum of expected values

            # If there are no legal actions ,return 0
            if numoflegalActions == 0:
                return  0
            
            return float(sumOfexpectvalues)/float(numoflegalActions)       
            

        # Get legal actions for Pacman at the root level.
        pacmanActions = gameState.getLegalActions(0)

        currScore = float('-inf')     # Initialize the current score to negative infinity.
        maxScoreAction = ''           # Initialize the variable to an empty string.

        for action in pacmanActions:
            # Generate successor state for Pacman's action.
            nextState = gameState.generateSuccessor(0,action)
            
            # Next level is a min level. (calling min for successors of the root)
            score = expectValue(nextState,0,1)

            # Choose the action with the maximum score.
            if score > currScore:
                maxScoreAction = action
                currScore = score

        return maxScoreAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Extract information from the game state
    PacmanPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()

    # newScaredTimes keeps the number of moves that ghosts remain scared when Pacman eats a power capsule.
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    
    # Calculate Manhattan distance to food
    from util import manhattanDistance
    foodList = Food.asList()
    foodDistance = [manhattanDistance(PacmanPos, pos) for pos in foodList]

    # Calculate Manhattan distance to ghosts
    ghostList = [ghost.getPosition() for ghost in GhostStates]
    ghostDistance = [manhattanDistance(PacmanPos, pos) for pos in ghostList]

    # Get the number of power Capsules
    numOfpowerCapsules = len(currentGameState.getCapsules())

    score = 0   # Initialize score 

    # Add components to the score
    score = score + currentGameState.getScore()    # Add current score
    score = score + 1.0 / (sum(foodDistance) + 1)  # Add the inverse of sum of food distances
    score = score + len(Food.asList(False))        # Add the number of food remaining

    # Adjust score based on ghost and capsules
    if sum(newScaredTimes) > 0:
        # There is remaining scared time for ghosts
        # Increase the score with the sum of scared times
        # Apply penalties for the number of power capsules and the sum of ghost distances
        score = score + sum(newScaredTimes) - numOfpowerCapsules - sum(ghostDistance)
    else:
        # There is no remaining scared time for ghosts
        # Increase the score with the sum of ghost distances and apply a bonus for the number of power capsules
        score = score - sum(ghostDistance) + numOfpowerCapsules
    
    # Return final score
    return score

# Abbreviation
better = betterEvaluationFunction
