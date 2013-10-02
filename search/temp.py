def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"
    "*** YOUR CODE HERE ***"

    frontier = []
    visited = []
    start = problem.getStartState()
    # [(int, int), "string", int]
    start = (start, "none", 0)
    # current_node = start
    frontier.insert(0, start)

    child_to_parent = {}
    while frontier:
        current_node = frontier.pop()
        children = problem.getSuccessors(current_node[0])
        for child in children:
            if child not in visited:
                child_to_parent[child] = current_node
                if problem.isGoalState(child[0]):
                    directions = []
                    path_node = child
                    directions.insert(0, path_node[1])
                    while path_node != start:
                        new_node = child_to_parent[path_node]
                        path_node = new_node
                        directions.insert(0,path_node[1])
                    directions.pop(0) # this is to get rid of the start "none" direction
                    print directions
                    return directions
                frontier.insert(0, child)
                visited.append(child)


    return []
