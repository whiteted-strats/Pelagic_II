# Markov chain for modelling the movement of the lone scientist in pelagic II
#   Outputs a graph showing the odds of a successful lure when arriving at a specific time.
# TODO take in actual console input using my lag analyser
# Clock, Data, State could be pulled out into separate scripts but it's nice to be all-in-one

import matplotlib.pyplot as plt

oneSecond = 60
oneMinute = 60 * oneSecond


class Clock:
    """
    Holds the update schedule, and provides conversion from ticks -> # updates
    Other key function is "advance"
    """

    def __init__(self, updateSchedule):
        assert updateSchedule[0] == 0
        self.timeI = 0
        self.updateSchedule = updateSchedule

    def advance(self):
        """
        Steps forward 1 update. Returns whether we're still running.
        NOTE states step from the new time to later times. Hence we stop when we reach our final time.
        """
        self.timeI += 1
        return self.timeI < len(self.updateSchedule) - 1

    def currTime(self):
        return self.updateSchedule[self.timeI]

    def finalTime(self):
        return self.updateSchedule[-1]

    def numUpdatesRemaining(self):
        return len(self.updateSchedule) - 1 - self.timeI

    def getNumUpdates(self, ticks):
        """
        Given a number of ticks, determine the number of updates that will be
        This is fundamental : updates are the atom

        If it overshoots the end of our update schedule, we extrapolate
          so that we'll be able to recover the progress to the nearest update
        """
        # No need for a binary search really

        currTime = self.currTime()
        finalTime = self.finalTime()
        endTime = currTime + ticks

        assert currTime < finalTime

        if endTime > finalTime:
            f = float(endTime - currTime) / (finalTime - currTime)
            return int(self.numUpdatesRemaining() * f)
        
        else:
            i = self.timeI
            while self.updateSchedule[i] < endTime:
                i += 1
            return i - self.timeI

class Data:
    """
    Never instanciated
    """

       # (prob, anim_length) pairs
    ANIMS = [(86 / 256, 395), (86 / 256, 465), (84 / 256, 440)]

    # MAP
    #		  3
    #		4		  5
    #
    #		   2
    #		6		  7
    #					 1

    # Main routes (front for 1 <-> 3)
    routes = {
	    (1,2)   :   (1,7,6,2),
	    (1,3)   :   (1,7,6,4,3),

	    (2,1)   :   (2,6,7,1),
	    (2,3)   :   (2,6,4,3),

	    (3,1)   :   (3,4,6,7,1),
	    (3,2)   :   (3,4,6,2),
    }

    # Alternate routes (around the back for 1 <-> 3)
    altRoutes = {
        (1,3)   : (1,7,5,4,3),
        (3,1)   : (3,4,5,7,1),
    }

    # Distances: Symmetry added below
    # Converted to ticks (slightly poor name?)
    dist = {
        # 'Spits' target -> rectangle
        (1,7) : 206.15,
        (3,4) : 213.59,
        (2,6) : 246.52,

        # Main rectangle
        (4,5) : 746.01,
        (5,7) : 346.01,
        (7,6) : 746.01,
        (6,4) : 346.01,
    }

    # Walk odds
    # Current pad -> [(target_pad, alternate?, prob)]
    # NOTE special case at 3, attempting to go to 3 causes a re-idle effectively (re-roll)
    #   -> makes animating at 3 more likely, interesting..
    #   Indicated by None values
    # NOTE where available, modelling 1/2 chance at alternate route
    walkDistrib = {
        1 : [(2, False, 172 / 256), (3, False, 42 / 256), (3, True, 42 / 256)],
        2 : [(1, False, 86 / 256), (3, False, 170 / 256)],
        3 : [(1, False, 43 / 256), (1, True, 43 / 256), (2, False, 86 / 256), (None, None, 84 / 256) ],
    }
    
    # Pad where each of 1,2,3 join the main rectangle. Common to alternate routes
    nextPadLeaving = {
        1 : 7,
        2 : 6,
        3 : 4,
    }


# Classic
_speed = 2.32

# Add symmetry to the distances
#   And convert to ticks
for s,d in list(Data.dist.keys()):
    dist = Data.dist[(s,d)]
    dist /= _speed
    Data.dist[(s,d)] = dist
    Data.dist[(d,s)] = dist

_speed = None


# Helper for if new
def add(acc, key, val):
    acc[key] = acc.get(key, 0) + val


class State:
    # Core state enum
    S_IDLE = 0
    S_ANIM = 1
    S_MOVE = 2
    S_ALT_MOVE = 3

 


    def __init__(self, coreState, pad, updatesRem, targetPad, origPad):
        """
        Core state as above.
        Pad is last one we were on
        Updates remaining until we change state
        Target pad used if moving
        Addition: original pad, else it's a pain to recover our route
        """

        self.coreState = coreState
        self.pad = pad
        self.updatesRem = updatesRem
        self.targetPad = targetPad
        self.origPad = origPad

    def addChildren(self, acc, clock, ownProb):
        """
        Determine our children with probabilities, and add them
        Currently keys are 5-tuples of initialisers, rather than State objects
        """
        if self.updatesRem > 1:
            add(
                acc,
                (self.coreState, self.pad, self.updatesRem - 1, self.targetPad, self.origPad),
                ownProb
            )
            return

        assert self.updatesRem == 1

        # If animating, become idle for an update
        if self.coreState == State.S_ANIM:
            assert self.pad in [1,2,3]
            add(acc, (State.S_IDLE, self.pad, 1, None, None), ownProb)
            return


        # If idle,
        if self.coreState == State.S_IDLE:

            #  1/2 chance to animate
            for animP, animLength in Data.ANIMS:
                lengthInUpdates = clock.getNumUpdates(animLength)
                add(acc,
                    (State.S_ANIM, self.pad, lengthInUpdates, None, None),
                    animP * 0.5 * ownProb
                )

            # 1/2 chance to pick a route
            for destPad, alt, prob in Data.walkDistrib[self.pad]:

                # Special 3->3 case, just idle
                if destPad is None:
                    assert self.pad == 3
                    add(acc, (State.S_IDLE, self.pad, 1, None, None), prob * 0.5 * ownProb)
                else:
                    moveState = State.S_ALT_MOVE if alt else State.S_MOVE
                    nextPad = Data.nextPadLeaving[self.pad]
                    distInUpdates = clock.getNumUpdates(Data.dist[(self.pad, nextPad)])
                    add(acc,
                        (moveState, self.pad, distInUpdates, destPad, self.pad),    # added original pad
                        prob * 0.5 * ownProb
                    )


            return


        # Else we're on our route, finishing a segment
        # 1. Get the route
        if self.coreState == State.S_MOVE:
            route = Data.routes[(self.origPad, self.targetPad)]
        else:
            assert self.coreState == State.S_ALT_MOVE
            route = Data.altRoutes[(self.origPad, self.targetPad)]
        assert route[-1] == self.targetPad
        
        # 2. Get the next 'wayspot' - pad
        routeIndex = route.index(self.pad)
        assert routeIndex != -1 and routeIndex <= len(route) - 2
        nextWayspot = route[routeIndex + 1]


        # 3. If we're arriving at our target, start idling
        #    Else start the next segment
        if nextWayspot == self.targetPad:
            add(acc, (State.S_IDLE, self.targetPad, 1, None, None), ownProb)
        
        else:
            secondWayspot = route[routeIndex + 2]
            distInUpdates = clock.getNumUpdates(Data.dist[(nextWayspot, secondWayspot)])
            add(acc,
                (self.coreState, nextWayspot, distInUpdates, self.targetPad, self.origPad),
                ownProb
            )





## ===================================================================
## ===================================================================

## For testing, assuming stable lag, 14 FPS
##   (7 drawn frames in 30 ticks (0.5s))
## TODO take these as input
clock = Clock(list(range(0, oneMinute + 1, 30)))



# State 5-Tuple -> probability
# Start by idling at pad 1. Tuple.
currDistrib = { (State.S_IDLE, 1, 1, None, None) : 1 }
pInLine = [0]
pGood = [0]
pStationaryAt3 = [0]
fourToFives = [0]
twoToOnes = [0]
altThreeToOnes = [0]

clock.timeI -= 1
while clock.advance():
    nextDistrib = dict()
    for t, p in currDistrib.items():
        state = State(*t)
        state.addChildren(nextDistrib, clock, p)

    currDistrib = nextDistrib
    
    weight = sum(currDistrib.values())
    assert 0.999 < weight < 1.001

    # Pulling out the probability that we're "good"
    # Now pretty accurate. Recall the map:
    #		  3
    #		4		  5
    #
    #		   2
    #		6		  7
    #					 1
    pg = 0
    fourToFive = 0
    threeAnim = 0
    twoToOne = 0
    altThreeToOne = 0
    pil = 0
    
    for t, p in currDistrib.items():
        s = State(*t)


        # If at 6, lone scientist must be moving -> 4, 2 or 7. All are good
        if s.pad == 6:
            assert s.coreState in [State.S_MOVE, State.S_ALT_MOVE] 
            pg += p
            if s.targetPad == 3:
                pil += p

        # If at 2, there is no choice of routes.
        # -> 1, will load and be bad.
        # Animating for >= 2s (conservative), fine
        # Else needs to not decide to -> 1
        if s.pad == 2:
            assert s.coreState != State.S_ALT_MOVE

            if s.coreState == State.S_MOVE:
                if s.targetPad == 1:
                    twoToOne += p
                else:
                    pg += p
                    pil += p

            else:
                if s.coreState == State.S_ANIM and s.updatesRem >= 3:
                    pg += p
                else:
                    # Idling or about to idle.

                    # 1/2 chance to anim. Further 2/3 chance to pick 3 as our target.
                    pg += (5/6) * p
                    twoToOne += (1/6) * p

                    pil += (2/6) * p

        # If at 4, must be moving, and only moving -> 5 is bad.
        # Interesting, if moving to 1, then when it loads it keeps it's target pad of 5 / 6.
        #   So moving -> 1 via the 'normal' route is a good state.
        if s.pad == 4:
            assert s.coreState in [State.S_MOVE, State.S_ALT_MOVE]
            if (s.coreState == State.S_ALT_MOVE and s.targetPad == 1):
                fourToFive += p
            else:
                pg += p

                if s.targetPad != 3:
                    pil += p
                
        # Being at 3 is pretty interesting.
        # Firstly moving 3 -> 3 is not redirected to 1 or 2, so is a 1 in 6 chance to reroll
        # Movement to 1 which was planned via 6 will Still go via 6 which is helpful.
        # If animating, we conservatively say there won't be time to reach 4.
        # If idle, we crunch the odds of choosing 2
        if s.pad == 3:
            if (s.coreState in [State.S_ALT_MOVE, State.S_MOVE]):
                # Allowing 3 -> 1 (normal route) actually adds a significant amount of weight
                if not (s.targetPad == 1 and s.coreState == State.S_ALT_MOVE):
                    pg += p
                else:
                    altThreeToOne += p
            elif s.coreState == State.S_ANIM:
                threeAnim += p
            else:
                # X = 1/6 + X/6, 6X = 1 + X, X = 1/5
                # -> 2 is good, but -> 1 will always be via 5? Since route is decided loaded.
                # Allowing inf. rerolls. Reasonable because script runs once per frame at this point.
                pg += (1/5) * p
                altThreeToOne += (1/5) * p

                pil += (1/5) * p


    pGood.append(pg)
    pInLine.append(pil)
    pStationaryAt3.append(threeAnim)
    fourToFives.append(fourToFive)
    twoToOnes.append(twoToOne)
    altThreeToOnes.append(altThreeToOne)


s = [x / 60 for x in clock.updateSchedule]

ax = plt.axes()
ax.yaxis.grid(True)
plt.plot(s, [x+y for x,y in zip(fourToFives, pGood)], color="xkcd:orange")   # Orange - 4 -> 5, needs strong lookup to get this.
plt.plot(s, pGood, color="xkcd:green")          # Green - the good scores
plt.plot(s, pInLine, color="xkcd:blue")         # Blue - the very good scores: guard will be in line with the door
#plt.plot(s, twoToOnes, color="xkcd:red")        # Red - 2 -> 1, bad because he avoids 6 when loaded.
#plt.plot(s, altThreeToOnes, color="xkcd:black") # Black - 3 -> 1 via 5, very bad.
plt.savefig("out", dpi=300)
plt.show()

print("Markov chain execution complete.")
