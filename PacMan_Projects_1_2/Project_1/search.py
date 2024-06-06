# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
   
    from util import Stack


    exploration_stack = Stack()                      # Create an empty stack (exploration_stack) and add the initial state
    exploration_stack.push((problem.getStartState(),[]))  # Start state and its path (tuples)

    visited = []                                     # Check when a state has been visited

    while not exploration_stack.isEmpty():
        # Get the state and path to the current node (from the stack)
        curr_state, curr_path = exploration_stack.pop()

        if problem.isGoalState(curr_state):
            # If the current state is the goal state, return the path
            return curr_path                

        if curr_state not in visited:
            visited.append(curr_state)                      # Mark the current state as visited
            successors = problem.getSuccessors(curr_state)  # Get the next states to explore (successor)

            for child, direction, _ in successors:
                # Push the child state and its updated path to the exploration stack
                exploration_stack.push((child, curr_path + [direction]))

    return None  # Return None if no path is found


    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue

    exploration_queue = Queue()             # Create an empty queue (exploration_queue) and add the initial state
    exploration_queue.push((problem.getStartState(), []))  # Start state and its path (tuples)

    visited = []                            # Check when a state has been visited

    while not exploration_queue.isEmpty():
        # Get the state and path to the current node (from the queue)
        current_state, current_path = exploration_queue.pop()

        
        if problem.isGoalState(current_state):
            # If the current state is the goal state, return the path
            return current_path

        # If the current state has not been visited
        if current_state not in visited:
            
            visited.append(current_state)                       # Mark the current state as visited
            successors = problem.getSuccessors(current_state)   # Get the next states to explore (successor)

            for next_state, action, _ in successors:
                # Create a new path by extending the current path
                new_path = current_path + [action]

                # Add the next state and its corresponding path to the queue
                exploration_queue.push((next_state, new_path))

    # If no path to the goal is found, return an empty list
    return []


    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

   

    from util import PriorityQueue

    exploration_prQueue = PriorityQueue()           # Create an empty priority queue (exploration_prQueue) and add the initial state
    exploration_prQueue.push((problem.getStartState(), [], 0),0)  # Start state, path and cost
    visited = []                                    # Check when a state has been visited


    while not exploration_prQueue.isEmpty():

        # Get the state, path and cost to the current node (from Priority Queue)
        currState, currPath, currCost = exploration_prQueue.pop()

        if problem.isGoalState(currState):
            # If the current state is the goal state, return the path
            return currPath

        # if the current state has not been visited
        if currState not in visited:

            visited.append(currState)                       # Mark the current state as visited
            successors = problem.getSuccessors(currState)   # Get the next states to explore (successor)

            for child, direction, step_cost in successors:

                if child not in visited:

                    # Calculate the new path and cost
                    new_path = currPath + [direction]
                    new_cost = currCost + step_cost
                    exploration_prQueue.push((child, new_path, new_cost),new_cost)

    # If no path to the goal is found, return an empty list
    return []  
    

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    exploration_prQueue = PriorityQueue()             # Create an empty priority queue (exploration_prQueue) and add the initial state
    start_state = problem.getStartState()
    exploration_prQueue.push((start_state, [], 0),0)  # State, path, and cost
    
    visited = []                                      # Check when a state has been visited


    while not exploration_prQueue.isEmpty():

        # Get the state, path and cost to the current node (from Priority Queue)
        current_state, current_path, current_cost = exploration_prQueue.pop()

        if problem.isGoalState(current_state):
            # If the current state is the goal state, return the path
            return current_path

        # if the current state has not been visited
        if current_state not in visited:

            visited.append(current_state)                       # Mark the current state as visited
            successors = problem.getSuccessors(current_state)   # Get the next states to explore (successor)


            for child, action, step_cost in successors:

                # Calculate the new path and cost
                new_path = current_path + [action]
                new_cost = current_cost + step_cost
                cost_to_go = new_cost + heuristic(child, problem)
                exploration_prQueue.push((child, new_path, new_cost), cost_to_go)

    return []

    
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
