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
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    #util.raiseNotDefined()
def printq(q):
    for x in q:
        print x
def breadthFirstSearch(problem):
    q=util.Queue()
    already_explored = []
    q.push([problem.getStartState(),[]]) 
    path = []
    while not q.isEmpty():
        currNode = q.pop()
        if currNode[0] not in already_explored: 
            already_explored.append(currNode[0])   
            for state, direction, price in problem.getSuccessors(currNode[0]):
                path=currNode[1][:] 
                path.append(direction)
                if problem.isGoalState(state):
                    return path
                else:
                    q.push([state, path])

    return path


def uniformCostSearch(problem):
    return aStarSearch(problem, nullHeuristic)


    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    distance = 0
    ##enodes keeps track of all the information of a given node
    ##Direction, Parent, Calculated heuristic, children, distance
    enodes = {}
    ##Checks to see what is and isn't a goalstate
    start = problem.getStartState()
    current = start
    enodes[current] = ['Startoo','Startoo', -1, [], distance]
    toCheck = []
    nlist = []
    nlist.append(current)

    while True:
        expanded = problem.getSuccessors(current)
        ##Increment distance for the algorithm
        distance = distance + 1
        ##Sets it to the program knows this node is expanded, will prevent any double expansions
        enodes[current][2] = -1

    ##Format the enodes dictionary and add the children nodes as well as various other information
        for nodes in expanded:
            if nodes[0] not in nlist:
                nlist.append(current)
                enodes[nodes[0]] = [nodes[1], current, distance + heuristic(nodes[0], problem), [], distance]
                enodes[current][3] += [nodes[0]]
                toCheck += [nodes[0]]

    ##Checks the nodes to see if they are the goal state, if not then expand another node
        for nodes in toCheck:
            nodes = toCheck.pop()
            #if nodes not in ngs:
        if problem.isGoalState(nodes):
            ##Generate the path from the goal node to the start node
            temp = []
            path = []
            while True:
                temp = enodes[nodes]
                if(temp[0] != 'Startoo'):
                    path.append(temp[0])
                else:
                    break
                nodes = temp[1]

            path.reverse()
            return path
        #else:
        #  ngs += [nodes]

    ##Get the best node for expansion
    ##Temporary variable to store the heuristic size
        temp = 0
        for nodes, stuff in enodes.iteritems():
            if(stuff[2] != -1):
                if temp == 0:
                    temp = stuff[2]
                    current = nodes
                    distance = stuff[4]
                elif stuff[2] < temp:
                    temp = stuff[2]
                    current = nodes
                    distance = stuff[4]

    return [w]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
