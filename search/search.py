# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):

    startState = problem.getStartState

    if problem.isGoalState(startState):
        return []


    fringe_set = []
    explored = []

    fringe_set.append((problem.getStartState(), '', 0))

    return dfs_helper(fringe_set, problem, explored)


def dfs_helper(fringe, problem, already_explored):
    node = fringe.pop()
    already_explored.append(node[0])

    if problem.isGoalState(node[0]):
        return[]

    for i in problem.getSuccessors(node[0]):
        if i[0] not in already_explored:
            fringe.append(i)
            path = dfs_helper(fringe, problem, already_explored)
            if (path != None):
                path.insert(0,i[1])
                return path

    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    #util.raiseNotDefined()
def printq(q):
    for x in q:
        print x
def breadthFirstSearch(problem):
    q=util.Queue()
    already_explored = []
    q.push([problem.getStartState(),[]]) 
    path = []
    retpath = path
    while not q.isEmpty():
        currNode = q.pop()
        if problem.isGoalState(currNode[0]):
            return currNode[1][:]

        if currNode[0] not in already_explored: 
            already_explored.append(currNode[0])
            slist = problem.getSuccessors(currNode[0])   
            for state, direction, price in slist:
                path=currNode[1][:] 
                path.append(direction)
                q.push([state, path])

    return retpath


def uniformCostSearch(problem):
    #Make the 3rd element our cost
    q = util.PriorityQueueWithFunction(lambda x: x[2])

    q.push((problem.getStartState(), [], 0))
    explored = []
    #explored = [problem.getStartState()]

    while not q.isEmpty():
        node = q.pop()
        if problem.isGoalState(node[0]):
            #print node
            return node[1]
        if node[0] not in explored:
            slist = problem.getSuccessors(node[0])
            for state, direction, price in slist:
                if state not in explored:
                    d = node[1] + [direction]
                    q.push((state, d, node[2]+price))
                explored.append(node[0])



    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    #Make the 3rd element our cost
    q = util.PriorityQueueWithFunction(lambda x: x[2] + heuristic(x[0], problem))

    q.push((problem.getStartState(), [], 0))
    explored = []
    #explored = [problem.getStartState()]

    while not q.isEmpty():
        node = q.pop()
        if problem.isGoalState(node[0]):
            #print node
            return node[1]
        if node[0] not in explored:
            slist = problem.getSuccessors(node[0])
            for state, direction, price in slist:
                if state not in explored:
                    d = node[1] + [direction]
                    q.push((state, d, node[2]+price))
                explored.append(node[0])



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
