# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def getClosestObject(position, objects):
    """
      Get closest object, be it a food pellet, or a ghost
    """
    if not objects: return None
    minDistance = 999999; minIndex = -1
    if len(objects) == 0: return 0
    for index, object in enumerate(objects):
        distance = util.manhattanDistance(position, object)
        if(distance < minDistance):
            minDistance = distance
            minIndex = index

    return objects[minIndex]

def capsuleExists(capsules, position):
    exists = False
    for capsule in capsules:
        if capsule == position: exists = True
    return exists

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        # ghosts = state.getGhostPositions()
        capsules = state.getCapsules()

        features = util.Counter()

        features["bias"] = 1.0

        ghostStates = state.getGhostStates()
        fleeingGhostsPositions = []
        ghosts = []

        for ghostState in ghostStates:
            if ghostState.scaredTimer > 0:
                fleeingGhostsPositions.append(ghostState.getPosition())
                features["eats-ghost"] = 1.0
            else:
                ghosts.append(ghostState.getPosition())

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        if ghosts:
            features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # count the number of fleeing ghosts 1-step away
        if fleeingGhostsPositions:
            features["#-of-fleeing-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in fleeingGhostsPositions)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        if not features["#-of-ghosts-1-step-away"] and capsuleExists(capsules, (next_x, next_y)):
            features["eats-capsule"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        closestCapsule = getClosestObject((x, y), capsules)
        if closestCapsule is not None: 
            closestCapsuleDistance = util.manhattanDistance((x, y), closestCapsule)
            features["closest-capsule"] = float(closestCapsuleDistance) / (walls.width * walls.height)

        # count the number of capsules 1-step away
        if capsules:
            features["#-of-capsules-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(c, walls) for c in capsules)


        closestFleeingGhost = getClosestObject((x, y), fleeingGhostsPositions)
        if closestFleeingGhost is not None: 
            closestFleeingGhostDistance = util.manhattanDistance((x, y), closestFleeingGhost)
            features["closest-fleeing-ghost"] = float(closestFleeingGhostDistance) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features