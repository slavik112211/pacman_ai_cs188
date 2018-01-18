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

class Node:
    def __init__(self, state, directions, cost):
        self.state = state
        self.directions = directions
        self.cost = cost

    def __eq__(self, other):
        return self.state == other.state

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

"""
1. python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
   Path found with total cost of 210 in 0.0 seconds
   Search nodes expanded: 549
2. python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=euclideanHeuristic
   Path found with total cost of 210 in 0.0 seconds
   Search nodes expanded: 557
3. python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=uniformCostSearch
   Path found with total cost of 210 in 0.0 seconds
   Search nodes expanded: 620

4. python pacman.py -l openMaze -z .5 -p StayEastSearchAgent
5. python pacman.py -l openMaze -z .5 -p StayWestSearchAgent
6. python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=breadthFirstSearch
   python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=aStarSearch,heuristic=euclideanHeuristic

7. python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
   Search nodes expanded: 1966
8. python pacman.py -l mediumCorners -p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic -z 0.5
   Search nodes expanded: 692
9. python pacman.py -l bigCorners -p SearchAgent -a fn=bfs,prob=CornersProblem -z 0.5
   Path found with total cost of 162 in 0.4 seconds
   Search nodes expanded: 7949
   Pacman emerges victorious! Score: 378
10.python pacman.py -l bigCorners -p SearchAgent -a fn=aStarSearch,prob=CornersProblem,heuristic=cornersHeuristic -z 0.5
   Path found with total cost of 162 in 0.1 seconds
   Search nodes expanded: 1726
   Pacman emerges victorious! Score: 378
   
11. python pacman.py -l trickySearch -p SearchAgent -a fn=bfs,prob=FoodSearchProblem -z 0.5
    Path found with total cost of 60 in 3.6 seconds
    Search nodes expanded: 16688
12. python pacman.py -l trickySearch -p AStarFoodSearchAgent
    Path found with total cost of 60 in 4.4 seconds
    Search nodes expanded: 6808
"""
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

def genericSearch(problem, frontier, search_type, heuristic=nullHeuristic):
    node = Node(problem.getStartState(), [], 0)
    explored_states = set()
    insertNodeIntoFrontier(node, problem, frontier, search_type, heuristic)

    # import pdb; pdb.set_trace()
    while not frontier.isEmpty():
        node = frontier.pop()
        explored_states.add(node.state)
        if problem.isGoalState(node.state): return node.directions
        for successor_node in problem.getSuccessors(node.state):
            if (successor_node[0] in explored_states) or \
                    (search_type == "bfs" and nodeInFrontier(successor_node[0], frontier)): continue
            next_node = Node(successor_node[0], node.directions + [successor_node[1]], node.cost + successor_node[2])
            insertNodeIntoFrontier(next_node, problem, frontier, search_type, heuristic)

    return [] # return empty directions history

def nodeInFrontier(node, frontier):
    nodes_in_frontier = list(map(lambda node: node.state, frontier.list))
    return node in nodes_in_frontier

def insertNodeIntoFrontier(node, problem, frontier, search_type, heuristic):
    if search_type == "ucs": frontier.update(node, node.cost)
    elif search_type == "A*": frontier.update(node, node.cost + heuristic(node.state, problem))
    else: frontier.push(node)

def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    frontier = util.Stack()
    return genericSearch(problem, frontier, 'dfs')

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()
    return genericSearch(problem, frontier, 'bfs')

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()
    return genericSearch(problem, frontier, 'ucs')

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue()
    # import cProfile; pr = cProfile.Profile(); pr.enable()
    path = genericSearch(problem, frontier, 'A*', heuristic)
    # pr.disable(); pr.dump_stats("searchStats")
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
