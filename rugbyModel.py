import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
import numpy as np
import seaborn as sns
from multiprocessing import Pool, cpu_count

################################################################################################################################
################################################################################################################################
################################################################################################################################

""" 
This class contains the STATE VARIABLES

Field Position:	The current location of play on the field. Represented as (x,y) coordinates
Ball position:	Indicates which player currently has possession of the ball
Ball coordinate:	The physical coordinates of the ball on the field (x,y)
Possession status:	Boolean variable indicating whether the offensive team retains possession
Sequence count:	The number of sequences elapsed
Defensive players:	A dictionary storing positions, average meters closed per phase, and speeds for defensive players
Offensive players:	A dictionary storing positions, average meters per carry, and speeds for offensive players
Defensive line average:	The average y-coordinate of all defensive players



"""

class RugbyModel:
    def __init__(self):
        #State variables
        self.state = {
            "field_position": (22, 15),  
            "ball_position": "9", 
            "ball_coordinate": (25, 15),
            "possession_status": True,
            "meters_gained": 0,
            "sequence_count": 0,
        }
        #Offensive player descriptions
        self.offensive_players = {
            "9": {"position": (25, 15), "avgMetre": 4, "avgSpeed": 5},
            "10": {"position": (28, 21), "avgMetre": 5, "avgSpeed": 6},
            "12": {"position": (28, 23), "avgMetre": 6, "avgSpeed": 6},
            "13": {"position": (31, 26), "avgMetre": 7, "avgSpeed": 6},
            "11": {"position": (31, 19), "avgMetre": 8, "avgSpeed": 7},
            "14": {"position": (29, 40), "avgMetre": 8, "avgSpeed": 7},
            "15": {"position": (32, 32), "avgMetre": 5, "avgSpeed": 6},
        }

        #Defensive player description
        self.defensive_players = {
            "7": {"position": (20, 17), "avgMetreClosed": 2, "avgSpeed": 5},  
            "9": {"position": (19, 14), "avgMetreClosed": 1.5, "avgSpeed": 5},  
            "10": {"position": (14, 22), "avgMetreClosed": 3, "avgSpeed": 5},  
            "12": {"position": (14, 26), "avgMetreClosed": 2.5, "avgSpeed": 6},  
            "13": {"position": (14, 30), "avgMetreClosed": 2.5, "avgSpeed": 6},  
            "11": {"position": (14, 34), "avgMetreClosed": 3, "avgSpeed": 6},  
            "14": {"position": (14, 15), "avgMetreClosed": 3, "avgSpeed": 6},  
            "15": {"position": (7, 20), "avgMetreClosed": 2, "avgSpeed": 5},  
        }
        
        self.initialPos = self.state["ball_coordinate"][1]
        
    def off_ball_movement(self, player, sequence, ball_position, actionStat):
        #Directs players movements according to Tango when they do not have the ball
        
        pitchW = 50  
        pitchL = 50 
        maxMov = 5 

        currentPos = self.offensive_players[player]["position"]
        newPos = list(currentPos)  

        #Players that have been involved in play are no longer involved in the set play and therefore no longer move
        #As this model does not explore opent play
        if actionStat.get(player, {}).get("didAction", False):
            return currentPos

        newPos[0] = max(currentPos[0], ball_position[0])

        #Makes sure players dont go infront of ball
        if currentPos[1] > ball_position[1]:
            newPos[1] = max(currentPos[1] + maxMov, ball_position[1] + 1)

        #Set movements off ball for players based on Tango
        if player == "12":
            if sequence > 1:
                newPos = [max(currentPos[0], ball_position[0] + 2), currentPos[1]]  

        elif player == "10":
            if self.state["ball_position"] == "12":
                newPos = [ball_position[0] + 3, ball_position[1] + 2]
                
            elif actionStat.get(player, {}).get("didAction", False):
                newPos = [max(currentPos[0], ball_position[0]), currentPos[1]]
                
        elif player == "11":
            if self.state["ball_position"] == "12":
                newPos = [self.offensive_players["10"]["position"][0]+1, 
                                self.offensive_players["10"]["position"][1] - 3]
            elif self.state["ball_position"] == "10":
                newPos = [self.offensive_players["10"]["position"][0] + 1, 
                                self.offensive_players["10"]["position"][1] + 6]
            elif actionStat.get(player, {}).get("didAction", False):
                newPos = [max(currentPos[0], ball_position[0]+1), currentPos[1]]

        elif player == "15":
            if actionStat.get(player, {}).get("didAction", False):
                newPos = [max(currentPos[0], ball_position[0]), currentPos[1]]
            else: 
                newPos = [max(currentPos[0], ball_position[0] + 3), ball_position[1] + 10]

        elif player == "14":
            if actionStat.get(player, {}).get("didAction", False):
                newPos = [max(currentPos[0], ball_position[0]), currentPos[1]]
            else: 
                newPos = [max(currentPos[0], ball_position[0] + 5), ball_position[1] + 20]

        elif player == "13":
            if self.state["ball_position"] == "12": 
                newPos = [ball_position[0]+1, ball_position[1] + 5]  
            elif self.state["ball_position"] == "10":
                newPos = [ball_position[0] - 1, ball_position[1] + 2]
            else:
                newPos = [max(currentPos[0], ball_position[0] + 2), ball_position[1] - 2]
                
        # Makes sure players dont move unrealistically
        distanceMoved = math.sqrt(
            (newPos[0] - currentPos[0]) ** 2 + (newPos[1] - currentPos[1]) ** 2
        )
        
        #Scale down if too much
        if distanceMoved > maxMov:
            scale = maxMov / distanceMoved
            newPos[0] = currentPos[0] + (newPos[0] - currentPos[0]) * scale
            newPos[1] = currentPos[1] + (newPos[1] - currentPos[1]) * scale
            
        elif player == "9":
            newPos = [max(currentPos[0], ball_position[0] + 2), ball_position[1]-3]

        # Keep players within in pitch
        newPos[0] = max(0, min(newPos[0], pitchL))
        newPos[1] = max(0, min(newPos[1], pitchW))

        return tuple(newPos)

      
    #Calculate the time it takes for a player to reach a target position
    @staticmethod
    def calcTime(start, end, speed):
        
        distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return distance / speed
    
    #Limit defender movements
    def limitDef(self, defender, targetPos, max_movement):

        current_x, current_y = defender["position"]
        target_x, target_y = targetPos
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)

        if distance > max_movement:
            scale = max_movement / distance
            target_x = current_x + (target_x - current_x) * scale
            target_y = current_y + (target_y - current_y) * scale

        print(f"Defender moving from ({current_x}, {current_y}) to ({target_x}, {target_y}) within {max_movement}m.")
        return (round(target_x, 1), round(target_y, 1))
    
    # Calculate the average y-position of all defenders
    def calcDefLineAvg(self,defensive_players):

        total_y = sum(player["position"][1] for player in defensive_players.values())
        num_defenders = len(defensive_players)
        return total_y / num_defenders if num_defenders > 0 else 0
    
    def maintainDefLn(self, defender, teammates, ball_position, defensive_line_y):
        spacing = 5 
        x, y = defender["position"]
        avg_y = defLnY


        new_x = max(x, ball_position[0] - 3)
        new_y = avg_y + spacing * (y - avg_y) / abs(y - avg_y) if y != avg_y else avg_y

        return (round(new_x, 1), round(new_y, 1))
       
    def idTarget(self, defPos, offensive_players):
        minD = float("inf")
        nextTarg = None

        for offensive_player, data in offensive_players.items():
            offensive_position = data["position"]
            distance = math.sqrt(
                (defPos[0] - offensive_position[0])**2 +
                (defPos[1] - offensive_position[1])**2
            )
            if distance < minD:
                minD = distance
                nextTarg = offensive_player

        return nextTarg
    
    def stopScrumWalkover(self, defPos, scrum_area):
        x, y = defPos
        scrum_x_start, scrum_y_start, scrum_width, scrum_height = scrum_area

        if scrum_x_start <= x <= scrum_x_start + scrum_width and scrum_y_start <= y <= scrum_y_start + scrum_height:
            if y < scrum_y_start: 
                return (x, scrum_y_start - 1)
            elif y > scrum_y_start + scrum_height: 
                return (x, scrum_y_start + scrum_height + 1)
            else:

                return (x, scrum_y_start - 1 if y < scrum_y_start + scrum_height / 2 else scrum_y_start + scrum_height + 1)

        return defPos

        """
        This is the TRANSITION FUNCTION
        
        """
    def transitionFunction(self, attackAction, defReaction):
        ballCarrier = attackAction["ballCarrier"]
        action = attackAction["action"]
        ballCarrierPos = self.offensive_players[ballCarrier].get("position")
        
        actionStat = {
            player: {"didAction": False} for player in self.offensive_players
        }
        
        if action in ["pass", "kick"]:
            actionStat[ballCarrier]["didAction"] = True
        
        
       #Defensive Reaction Transition
        for defender, reaction in defReaction.items():
            defPos = self.defensive_players[defender]["position"]
            targetPos = reaction["target"]
            distToBall = reaction["distance"]
            currentPos = self.defensive_players[defender]["position"]
            
            if distToBall < 1:
                self.state["possession_status"] = False
                print(f"{ballCarrier} was tackled by {defender} (close-range override).")
                return {"tackled": True}

            limitedPos = self.limitDef(self.defensive_players[defender],targetPos,self.defensive_players[defender]["avgMetreClosed"])
            self.defensive_players[defender]["position"] = limitedPos

            if reaction["action"] == "bite":
                if random.random() < reaction["tackleProba"]:
                    self.state["possession_status"] = False
                    print(f"{ballCarrier} was tackled by {defender}.")
                    return {}
        
        #Offensive Action Transition
        if action in ["carry", "kick", "pass"]:
            for defender, data in self.defensive_players.items():
                distance = math.sqrt(
                    (data["position"][0] - ballCarrierPos[0])**2 +
                    (data["position"][1] - ballCarrierPos[1])**2
                )
                if distance < 0.5:
                    self.state["possession_status"] = False
                    print(f"{ballCarrier} was tackled by {defender} at a distance of {distance:.2f} meters.")
                    return {}
                

        if action == "carry":
            tackleProba = defReaction[ballCarrier]["tackleProba"]
            distance = defReaction[ballCarrier]["distance"]

            if distance <= 1:
                tackleProba = 0.95

            if random.random() < tackleProba:
                self.state["possession_status"] = False
                tackler = max(
                    defReaction.keys(),
                    key=lambda defender: defReaction[defender]["tackleProba"]
                )
                print(f"{ballCarrier} was tackled by {tackler} at a distance of {distance:.2f} meters.")
                return {}
            else:
                meters = self.offensive_players[ballCarrier]["avgMetre"]
                self.state["meters_gained"] += meters
                newPos = (ballCarrierPos[0] - meters, ballCarrierPos[1])
                self.offensive_players[ballCarrier]["position"] = newPos
                self.state["ball_coordinate"] = newPos
            return {}

        elif action == "pass":
            target = attackAction["target"]
            self.state["ball_position"] = target
            targetPos = self.offensive_players[target]["position"]
            self.offensive_players[target]["position"] = (
                ballCarrierPos[0] + 1, targetPos[1]
            )
            self.state["ball_coordinate"] = self.offensive_players[target]["position"] 
            print(f"{ballCarrier} passed to {target} at {self.offensive_players[target]['position']}")
            return {}

        elif action == "kick":
            target = attackAction["target"]
            kickType = attackAction.get("kickType", "grubber") 
            ballCarrierPos = self.offensive_players[ballCarrier]["position"]
            
            if kickType == "grubber":
                kickdist = random.randint(5, 20) 

            elif kickType == "crossfield":
                kickdist = random.randint(10, 30) 

            if target in self.offensive_players:
                targetPos = self.offensive_players[target]["position"]
            else:
                
                if target == "14":
                    targetPos = (ballCarrierPos[0] - kickdist, 45)
                elif target == "11": 
                    targetPos = (ballCarrierPos[0] - kickdist, 30)
                else:
                    
                    targetPos = (ballCarrierPos[0] - kickdist, ballCarrierPos[1])

          
            direction_x = targetPos[0] - ballCarrierPos[0]
            direction_y = targetPos[1] - ballCarrierPos[1]

            
            distTarget = math.sqrt(direction_x**2 + direction_y**2)
            normDirect = (direction_x / distTarget, direction_y / distTarget)

   
            landingPos = (
                ballCarrierPos[0] + normDirect[0] * kickdist,
                ballCarrierPos[1] + normDirect[1] * kickdist,
            )
            

            if target:
                target_speed = self.offensive_players[target].get("avgSpeed", 5)  
                timeReach = self.calcTime(targetPos, landingPos, targetSpeed)

                if timeReach <= kickdist / 10:  

                    self.state["ball_position"] = target
                    self.offensive_players[target]["position"] = landingPos
                    self.state["ball_coordinate"] = landingPos
                    print(f"{ballCarrier} kicked toward {target} who gathered the ball.")
                else:    

                    for defender, defenderData in self.defensive_players.items():
                        defPos = defenderData["position"]
                        defender_speed = defenderData.get("avgSpeed", 4)
                        timeReach_defender = self.calcTime(defPos, landingPos, defender_speed)

                        if timeReach_defender <= kickdist / 10:
                            self.state["possession_status"] = False
                            self.state["ball_position"] = None
                            self.state["ball_coordinate"] = landingPos
                            print(f"{ballCarrier} kicked toward {target}, but {defender} gathered the ball.")
                            return {"landingPos": landingPos, "target": target}
                        else: 

                            closest_player = None
                            closest_distance = float("inf")
                            closest_team = None

                            for player, data in self.offensive_players.items():
                                distance = math.sqrt(
                                    (data["position"][0] - landingPos[0]) ** 2 +
                                    (data["position"][1] - landingPos[1]) ** 2
                                )
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_player = player
                                    closest_team = "offense"


                            for defender, defenderData in self.defensive_players.items():
                                distance = math.sqrt(
                                    (defenderData["position"][0] - landingPos[0]) ** 2 +
                                    (defenderData["position"][1] - landingPos[1]) ** 2
                                )
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_player = defender
                                    closest_team = "defense"

                            if closest_team == "offense":
                                self.state["ball_position"] = closest_player
                                self.state["ball_coordinate"] = landingPos
                                self.offensive_players[closest_player]["position"] = landingPos
                                print(f"{ballCarrier} kicked toward {target}, but {closest_player} (offense) gathered the ball.")
                                return {"landingPos": landingPos, "target": closest_player}
                            elif closest_team == "defense":
                                self.state["possession_status"] = False
                                self.state["ball_position"] = landingPos
                                self.state["ball_coordinate"] = landingPos
                                print(f"{ballCarrier} kicked toward {target}, but {closest_player} (defense) gathered the ball. Turnover!")
                                return {"landingPos": landingPos, "target": closest_player}
                                                
        
        for player, data in self.offensive_players.items():
            if player != self.state["ball_position"]:  
                newPos = self.off_ball_movement(
                    player,
                    self.state["sequence_count"],
                    self.offensive_players[self.state["ball_position"]]["position"],
                    actionStat
                )
                self.offensive_players[player]["position"] = newPos

        return {}
    
    """
    OBJECTIVE FUNCTION
    """
    
    def objective_function(self):
        final_y_position = self.state["ball_coordinate"][1]
        vertical_distance = abs(final_y_position - self.initialPos)
        return vertical_distance
    
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

class OffensivePolicy:
    def __init__(self):
        self.offensive_probabilities = {
             "9": [
                {"action": "pass", "target": "12", "probability": 0.92, 
                 "condition": lambda state, reactions: state["field_position"][0] > 20},
                {"action": "pass", "target": "10", "probability": 0.05, 
                 "condition": lambda state, reactions: reactions["10"]["action"] in ["hover", "bite"]},
                {"action": "carry", "probability": 0.03, 
                 "condition": lambda state, reactions: all(reactions[defender]["distance"] > 3 for defender in reactions)},
            ],
            "12": [
                {"action": "pass", "target": "10", "probability": 0.8, 
                 "condition": lambda state, reactions: reactions["13"]["action"] != "hover" or reactions["10"]["action"] in ["hover", "bite"]},
                {"action": "pass", "target": "13", "probability": 0.2, 
                 "condition": lambda state, reactions: reactions["13"]["action"] == "swim"},
                {"action": "carry", "probability": 0.01, 
                 "condition": lambda state, reactions: all(reactions[defender]["distance"] > 3 for defender in reactions)},
            ],
            "10": [
                {"action": "pass", "target": "11", "probability": 0.4, 
                 "condition": lambda state, reactions: reactions["11"]["action"] == "hover" or reactions["10"]["distance"] > 2},
                {"action": "kick", "target": "14", "probability": 0.3, 
                 "condition": lambda state, reactions: reactions["15"]["action"] in ["hover", "stay_deep"]},
                {"action": "carry", "probability": 0.3, 
                 "condition": lambda state, reactions: reactions["10"]["action"] != "bite" and reactions["15"]["distance"] > 3},
            ],
            "13": [
                {"action": "carry", "probability": 0.1, 
                 "condition": lambda state, reactions: reactions["13"]["distance"] > 3},
            ],
            "11": [
                {"action": "carry", "probability": 0.6, 
                 "condition": lambda state, reactions: reactions["11"]["distance"] > 2 and reactions["15"]["action"] == "hover"},
                {"action": "pass", "target": "15", "probability": 0.3, 
                 "condition": lambda state, reactions: reactions["15"]["action"] in ["hover", "bite"]},
                {"action": "kick", "target": "14", "probability": 0.1, 
                 "condition": lambda state, reactions: reactions["14"]["action"] == "hover"},
            ],
            "15": [
                {"action": "carry", "probability": 0.1, 
                "condition": lambda state, reactions: all(reactions[defender]["distance"] > 3 for defender in reactions)},
                {"action": "pass", "target": "14", "probability": 0.7, 
                "condition": lambda state, reactions: reactions["14"]["distance"] > 2 and reactions["15"]["action"] != "bite"},
                {"action": "kick", "target": "14", "probability": 0.2, 
                "condition": lambda state, reactions: reactions["14"]["distance"] > 4},
            ],
            "14": [
                {"action": "carry", "probability": 1.0, 
                 "condition": lambda state, reactions: reactions["14"]["distance"] > 2},
            ],
        }

    def decide(self, state, defReaction, offensive_players):
      
        ballCarrier = state["ball_position"]
        possible_actions = self.offensive_probabilities.get(ballCarrier, [])
        valid_actions = []

        for action in possible_actions:
            if action["condition"](state, defReaction):
                adjusted_probability = action["probability"]


                target = action.get("target")
                if target:
                    if defReaction[target]["distance"] <= 2:
                        adjusted_probability *= 0.8 
                    elif defReaction[target]["distance"] > 4:
                        adjusted_probability *= 1.2  

                if defReaction[ballCarrier]["distance"] <= 2:
                    adjusted_probability *= 0.8  
                elif defReaction[ballCarrier]["distance"] > 4:
                    adjusted_probability *= 1.2  

                valid_actions.append({"action": action, "adjusted_probability": adjusted_probability})


        if not valid_actions:
            return {"action": "carry", "ballCarrier": ballCarrier}

        total_prob = sum(a["adjusted_probability"] for a in valid_actions)
        normalized_actions = [(a["action"], a["adjusted_probability"] / total_prob) for a in valid_actions]
        selected_action = random.choices(
            [a[0] for a in normalized_actions],
            weights=[a[1] for a in normalized_actions]
        )[0]


        decision = {"action": selected_action["action"], "ballCarrier": ballCarrier}
        if "target" in selected_action:
            decision["target"] = selected_action["target"]
        return decision

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


class DefensivePolicy:
    def __init__(self, model):
        self.model = model
        self.defensive_probabilities = {
            "9": {"primary_target": "9", "base_action": "stay_near_scrum"},
            "10": {"primary_target": "10", "base_action": "hover"},
            "11": {"primary_target": "15", "base_action": "hover"},
            "12": {"primary_target": "12", "base_action": "hover"},
            "13": {"primary_target": "13", "base_action": "hover"},
            "14": {"primary_target": "11", "base_action": "hover"},
            "7": {"primary_target": "10", "base_action": "hover"},
        }
    def probabilistic_defReaction(self, defender, ballCarrier, ballCarrierPos, defenders):
        """
        Determine the action for the defender based on probabilities.
        """
        distance = math.sqrt(
            (defender["position"][0] - ballCarrierPos[0])**2 +
            (defender["position"][1] - ballCarrierPos[1])**2
        )
        if distance <= 3:
            return random.choices(["hover", "bite"], weights=[0.3, 0.7])[0]
        elif distance > 3:
            return random.choices(["hover", "swim"], weights=[0.7, 0.3])[0]
        return "hover"

    def swim_off_to_nextTarg(self, defPos, offensive_players, defensive_line_avg,current_target=None):
        closest_player = None
        closest_distance = float('inf')

        for player, data in offensive_players.items():
            distance = math.sqrt(
                (defPos[0] - data["position"][0])**2 +
                (defPos[1] - data["position"][1])**2
            )
            if distance < closest_distance:
                closest_distance = distance
                closest_player = player  


        if closest_player is None:
            print("No offensive player found for swim action. Maintaining current target.")
            if current_target is not None:
                return current_target  
            else:
                return defPos  


        targetPos = offensive_players[closest_player]["position"]
        targetPos = (
            targetPos[0], 
            defensive_line_avg  
        )

        return closest_player
    
    def decide(self, state, attackAction, offensive_players, defensive_players):
        reactions = {}
        ballCarrier = attackAction["ballCarrier"]
        ballCarrierPos = offensive_players[attackAction["ballCarrier"]]["position"]
        
        if isinstance(ballCarrier, tuple):
            ballCarrierPos = ballCarrier
        else:
            ballCarrierPos = offensive_players[ballCarrier]["position"]
        
        for defender, data in defensive_players.items():
            reactions[defender] = {}

        for defender, data in defensive_players.items():
            defPos = data["position"]


            action = self.probabilistic_defReaction(data, ballCarrier, ballCarrierPos, defensive_players.values())
            defensive_line_avg = self.model.calcDefLineAvg(defensive_players)
            tackleProba = 0.1
            if action == "swim":
                current_target = reactions[defender].get("target")
                nextTarg = self.swim_off_to_nextTarg(defPos, offensive_players,defensive_line_avg, current_target)
                targetPos = offensive_players[nextTarg]["position"]
                tackleProba = 0.2
                
            elif action == "hover":
                targetPos = self.model.maintainDefLn(
                    data, list(defensive_players.values()), ballCarrierPos, defensive_line_avg
                )
                tackleProba = 0.35
                
            elif action == "bite":
                targetPos = ballCarrierPos  


                distance = math.sqrt(
                    (defPos[0] - targetPos[0]) ** 2 +
                    (defPos[1] - targetPos[1]) ** 2
                )
                
                tackleProba == 0.95 if distance <= 1 else tackleProba == 0.4
            
            if defender == "15": 
                targetPos = (max(ballCarrierPos[0] - 10, 20), max(ballCarrierPos[1]- 3, 20))       
            
            
            increment_x = (targetPos[0] - defPos[0]) * 0.5 
            increment_y = (targetPos[1] - defPos[1]) * 0.5
            targetPos = (defPos[0] + increment_x, defPos[1] + increment_y)

                     

            reactions[defender] = {
                "action": action,
                "target": targetPos,
                "distance": math.sqrt(
                    (defPos[0] - ballCarrierPos[0])**2 +
                    (defPos[1] - ballCarrierPos[1])**2
                ),
                "tackleProba": tackleProba,
            }
             
            
        return reactions
    
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

class RugbySimulation:
    def __init__(self, model, offensive_policy, defensive_policy, visualizer=None):
        self.model = model
        self.offensive_policy = offensive_policy
        self.defensive_policy = defensive_policy
        self.visualizer = visualizer
        
    def run_simulation(self,logger, visualize=True):
        """
        Runs the simulation until a try is scored, a player is tackled, or possession is lost.
        """
        print("\nSimulation Start:\n")
        
        if visualize and self.visualizer:
            print("Rendering starting positions...")
            self.visualizer.render()

        while True:
            sequence = self.model.state["sequence_count"]

            if not self.model.state["possession_status"]:
                print(f"Turnover: Simulation ends at sequence {sequence}.")
                if visualize and self.visualizer:
                    self.visualizer.render()
                break

            defReaction = self.defensive_policy.decide(
                self.model.state,
                {"ballCarrier": self.model.state["ball_position"]},
                self.model.offensive_players,
                self.model.defensive_players,
            )
            print(f"sequence {sequence}: Defense - {defReaction}")

            attackAction = self.offensive_policy.decide(self.model.state, defReaction, self.model.offensive_players)

            transition_data = self.model.transitionFunction(attackAction, defReaction)
            
            if visualize and self.visualizer:
                
                if "landingPos" in transition_data:
                    self.visualizer.add_landing_marker(transition_data["landingPos"])
                if "target" in transition_data:
                    self.visualizer.highlight_player(transition_data["target"], "green")
                self.visualizer.render()
                
            logger.log_sequence(
                self.model.state,
                decisions=[attackAction],
                ball_position=self.model.state.get("ball_position"),
                defensive_positions=[data["position"] for data in self.model.defensive_players.values()],
            )
                
            if isinstance(self.model.state["ball_position"], tuple):
                print(f"Ball is at rest at {self.model.state['ball_position']}.")
            
            if self.model.state["meters_gained"] >= 22:
                print("Try scored! Simulation ends.")
                if visualize and self.visualizer:
                    self.visualizer.render()
                break

            if self.model.state.get("tackled", False):
                print(f"Player tackled! Simulation ends at sequence {sequence}.")
                if visualize and self.visualizer:
                    self.visualizer.render()
                break

            self.model.state["sequence_count"] += 1
            
        print(f"Total Meters Gained (y-axis): {self.model.objective_function()} meters")
        
        
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
    
class RugbyVisualizer:
    def __init__(self, model):
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(8, 12))

        self.draw_field()

        self.offensive_markers = {}
        self.defensive_markers = {}
        self.offensive_labels = {}
        self.defensive_labels = {}
        self.ball_marker = None
        self.initialize_markers()

    def draw_field(self):
        """Draw the rugby field."""
        self.ax.set_xlim(0, 50)  
        self.ax.set_ylim(50, 0)  
        self.ax.set_aspect('equal', adjustable='box')


        self.ax.axhline(22, color='black', linestyle='--', linewidth=1, label="22m Line")
        self.ax.axhline(0, color='red', linestyle='-', linewidth=2, label="Try Line")


        scrum_rect = patches.Rectangle(
        (15 - 1.5, 22 - 3,),  
        3,  
        6,  
        color='gray',
        alpha=0.5
        )
        self.ax.add_patch(scrum_rect)

        self.ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        self.ax.set_xticks(range(0, 51, 5))
        self.ax.set_yticks(range(0, 51, 5))
        self.ax.legend()
    
    def add_landing_marker(self, position):
        """Add a marker for the ball's landing spot."""
        self.ax.add_patch(patches.Circle(position[::-1], 0.7, color='orange', alpha=0.6))

    def highlight_player(self, player, color):
        """Highlight a player by changing their marker color temporarily."""
        if player in self.offensive_markers:
            self.offensive_markers[player].set_facecolor(color)
        elif player in self.defensive_markers:
            self.defensive_markers[player].set_facecolor(color)
        self.fig.canvas.draw_idle()
        plt.pause(0.5)  

    def initialize_markers(self):
        """Create initial markers and text labels for players and the ball."""
        for player, data in self.model.offensive_players.items():
            position = data["position"]
            self.offensive_markers[player] = self.ax.add_patch(
                patches.Circle(position[::-1], 1, color='green', alpha=0.7)
            )
            self.offensive_labels[player] = self.ax.text(
                *position[::-1], player, ha='center', va='center', color='black'
            )

        for player, data in self.model.defensive_players.items():
            position = data["position"]
            self.defensive_markers[player] = self.ax.add_patch(
                patches.Circle(position[::-1], 1, color='blue', alpha=0.7)
            )
            self.defensive_labels[player] = self.ax.text(
                *position[::-1], player, ha='center', va='center', color='black'
            )

        ball_position = self.model.offensive_players[self.model.state["ball_position"]]["position"]
        self.ball_marker = self.ax.add_patch(
            patches.Circle(ball_position[::-1], 0.5, color='yellow', alpha=0.9)
        )

    def update_positions(self):
        for player, marker in self.offensive_markers.items():
            position = self.model.offensive_players[player]["position"]
            marker.center = position[::-1]  
            self.offensive_labels[player].set_position(position[::-1]) 

        for player, marker in self.defensive_markers.items():
            position = self.model.defensive_players[player]["position"]
            marker.center = position[::-1]
            self.defensive_labels[player].set_position(position[::-1])

        ball_position = self.model.state["ball_position"]
        if isinstance(ball_position, tuple): 
            self.ball_marker.center = ball_position[::-1]
        elif ball_position in self.model.offensive_players: 
            position = self.model.offensive_players[ball_position]["position"]
            self.ball_marker.center = position[::-1]
        else:
            raise ValueError(f"Invalid ball position: {ball_position}")

    def render(self):
        """Render the visualization for the current state."""
        self.update_positions()
        self.fig.canvas.draw_idle()
        plt.pause(15)
  
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

   
class RugbyEvaluator:
    def __init__(self):
        self.simulation_data = []
        self.best_sequence = None 
        self.best_score = float('-inf')

    def log_simulation(self, simulation_metrics):
        """Log the results of a single simulation."""
        self.simulation_data.append(simulation_metrics)

        score = simulation_metrics["meters_gained"]  
        if simulation_metrics["try_scored"]:  
            score += 5

        if score > self.best_score:
            self.best_score = score
            self.best_sequence = simulation_metrics["decision_log"]
    
    def get_best_sequence(self):
        """Return the best sequence and its score."""
        return self.best_sequence, self.best_score

    def save_to_csv(self, filename="simulation_results.csv"):
        """Save the simulation data to a CSV file with separate columns for action probabilities."""
        rows = []
        for data in self.simulation_data:
            decision_counts = {"pass": 0, "carry": 0, "kick": 0}
            for decisions in data["decision_log"]:
                for decision in decisions:
                    decision_counts[decision["action"]] += 1

            total_decisions = sum(decision_counts.values())
            action_probabilities = {
                k: v / total_decisions for k, v in decision_counts.items()
            } if total_decisions > 0 else {"pass": 0, "carry": 0, "kick": 0}

            rows.append({
                "meters_gained": data["meters_gained"],
                "sequence_count": data["sequence_count"],
                "try_scored": data["try_scored"],
                "pass_probability": action_probabilities["pass"],
                "carry_probability": action_probabilities["carry"],
                "kick_probability": action_probabilities["kick"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Simulation data saved to {filename}")
        
    def calculate_action_probabilities(self):

        decision_counts = {"pass": 0, "carry": 0, "kick": 0}
        for data in self.simulation_data:
            for decisions in data["decision_log"]:
                for decision in decisions:
                    decision_counts[decision["action"]] += 1

        total_decisions = sum(decision_counts.values())
        return {k: v / total_decisions for k, v in decision_counts.items()} if total_decisions > 0 else {}
 
    def save_metrics_to_txt(self, filename="simulation_metrics.txt"):

        try_count = sum(1 for data in self.simulation_data if data["try_scored"])
        total_simulations = len(self.simulation_data)
        try_conversion_rate = (try_count / total_simulations) * 100 if total_simulations > 0 else 0
        overall_action_probabilities = self.calculate_action_probabilities()
        best_sequence, best_score = self.get_best_sequence()
        with open(filename, "w") as f:
            f.write(f"Simulation Metrics\n")
            f.write(f"===================\n")
            f.write(f"Total Simulations: {total_simulations}\n")
            f.write(f"Try Conversion Rate: {try_conversion_rate:.2f}%\n")
            f.write(f"Overall Decision Probabilities:\n")
            for decision, probability in overall_action_probabilities.items():
                f.write(f"  {decision.capitalize()}: {probability:.2f}\n")
            f.write(f"\nBest Sequence and Score:\n")
            f.write(f"  Best Sequence: {best_sequence}\n")
            f.write(f"  Best Score: {best_score:.2f}\n")
        print(f"Metrics saved to {filename}")

    def load_from_csv(self, filename):
        self.simulation_data = pd.read_csv(filename).to_dict(orient="records")

    def meters_gained_distribution(self):
        meters = [data["meters_gained"] for data in self.simulation_data]
        plt.hist(meters, bins=20, edgecolor="black")
        plt.title("Meters Gained Distribution")
        plt.xlabel("Meters Gained")
        plt.ylabel("Frequency")
        plt.show()
        
        plt.scatter(range(1, len(meters) + 1), meters, color="blue", alpha=0.7)
        plt.title("Meters Gained per Simulation (Scatter Plot)")
        plt.xlabel("Simulation Number")
        plt.ylabel("Meters Gained")
        plt.show()
        
    def meters_gained_per_sequence(self):
        for idx, data in enumerate(self.simulation_data):
            meters_per_sequence = data.get("meters_per_sequence", [])
            plt.plot(range(1, len(meters_per_sequence) + 1), meters_per_sequence, marker="o", label=f"Sim {idx + 1}")

        plt.title("Meters Gained per sequence (All Simulations)")
        plt.xlabel("sequence Number")
        plt.ylabel("Meters Gained")
        plt.legend()
        plt.show()

    def try_conversion_rate(self):
        try_count = sum(1 for data in self.simulation_data if data["try_scored"])
        total = len(self.simulation_data)
        conversion_rate = (try_count / total) * 100 if total > 0 else 0
        print(f"Try Conversion Rate: {conversion_rate:.2f}%")

    def decision_frequency(self):
        decisions_per_sequence = {}

        for data in self.simulation_data:
            for sequence, decisions in enumerate(data["decision_log"]):
                for decision in decisions:
                    decision_key = (decision["action"], decision.get("target", "N/A"))
                    if decision_key not in decisions_per_sequence:
                        decisions_per_sequence[decision_key] = []
                    decisions_per_sequence[decision_key].append(sequence + 1)

        for decision, sequences in decisions_per_sequence.items():
            plt.scatter(sequences, [str(decision)] * len(sequences), alpha=0.7, label=str(decision))

        plt.title("Decision Frequency by sequence")
        plt.xlabel("sequence Number")
        plt.ylabel("Decisions")
        plt.legend()
        plt.show()

    def ball_position_trajectory(self):
        for idx, data in enumerate(self.simulation_data):
            if "ball_positions" in data:
                positions = np.array(data["ball_positions"])
                plt.plot(positions[:, 1], positions[:, 0], marker="o", label=f"Sim {idx + 1}")
        
        plt.title("Ball Position Trajectory")
        plt.xlabel("Y Coordinate")
        plt.ylabel("X Coordinate")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
        
    def player_decision_probabilities(self):
        decisions = []
        for sim in self.simulation_data:
            for sequence_decisions in sim["decision_log"]:
                for decision in sequence_decisions:
                    decisions.append(decision["action"])

        sns.countplot(y=decisions, order=pd.Series(decisions).value_counts().index)
        plt.title("Player Decision Probabilities")
        plt.xlabel("Frequency")
        plt.ylabel("Decision Type")
        plt.show()
        

    def defensive_action_distribution(self):
        actions = []
        for sim in self.simulation_data:
            for defender, reaction in sim["defensive_positions"]:
                actions.append(reaction.get("action", "unknown"))

        sns.countplot(y=actions, order=pd.Series(actions).value_counts().index)
        plt.title("Defensive Action Distribution")
        plt.xlabel("Frequency")
        plt.ylabel("Action Type")
        plt.show()

    def ball_position_heatmap(self):
        ball_positions = []
        for sim in self.simulation_data:
            ball_positions.extend(sim["ball_positions"])

        positions = np.array(ball_positions)
        plt.hexbin(positions[:, 1], positions[:, 0], gridsize=30, cmap="Blues", extent=(0, 50, 0, 50))
        plt.colorbar(label="Frequency")
        plt.title("Ball Position Heatmap")
        plt.xlabel("Y Coordinate")
        plt.ylabel("X Coordinate")
        plt.gca().invert_yaxis()
        plt.show()

    def decision_success_rate(self):
        success_counts = {"pass": 0, "kick": 0, "carry": 0}
        total_counts = {"pass": 0, "kick": 0, "carry": 0}

        for sim in self.simulation_data:
            for sequence_decisions in sim["decision_log"]:
                for decision in sequence_decisions:
                    action = decision["action"]
                    total_counts[action] += 1
                    if decision.get("success", False):
                        success_counts[action] += 1

        success_rates = {key: success_counts[key] / total_counts[key] if total_counts[key] > 0 else 0
                         for key in total_counts.keys()}
        sns.barplot(x=list(success_rates.keys()), y=list(success_rates.values()))
        plt.title("Decision Success Rates")
        plt.xlabel("Action Type")
        plt.ylabel("Success Rate")
        plt.show()

    def transition_probability_matrix(self):
        transitions = {}
        for sim in self.simulation_data:
            previous_state = None
            for sequence, decisions in enumerate(sim["decision_log"]):
                current_state = decisions[0]["action"] 
                if previous_state is not None:
                    if previous_state not in transitions:
                        transitions[previous_state] = {}
                    if current_state not in transitions[previous_state]:
                        transitions[previous_state][current_state] = 0
                    transitions[previous_state][current_state] += 1
                previous_state = current_state

        df = pd.DataFrame(transitions).fillna(0)
        for col in df.columns:
            df[col] = df[col] / df[col].sum()
        sns.heatmap(df, annot=True, cmap="Blues")
        plt.title("State Transition Probabilities")
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.show()

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

class SimulationLogger:
    def __init__(self):
        self.meters_gained = 0
        self.sequence_count = 0
        self.try_scored = False
        self.decision_log = []  
        self.ball_positions = [] 
        self.defensive_positions = []
        self.meters_per_sequence = [] 

    def log_sequence(self, state, decisions, ball_position, defensive_positions):
        current_meters = state["meters_gained"] 
        sequence_meters = current_meters - sum(self.meters_per_sequence)
        self.meters_per_sequence.append(sequence_meters) 
        self.meters_gained = state["meters_gained"]
        self.sequence_count = state["sequence_count"]
        self.try_scored = state["meters_gained"] >= 22
        
        for decision in decisions:
            decision["success"] = self.evaluate_decision_success(decision, state, defensive_positions)

        
        self.decision_log.append(decisions)
        self.ball_positions.append(state["ball_coordinate"])
        self.defensive_positions.append(defensive_positions)
        
    @staticmethod
    def evaluate_decision_success(decision, state, defensive_positions):
            if decision["action"] == "pass":
                return "target" in decision and decision["target"] in state["ball_position"]
            elif decision["action"] == "carry":
                return state["possession_status"]
            elif decision["action"] == "kick":
                return state["possession_status"] and state["ball_position"] != "turnover"
            return False

    def get_metrics(self):
        return {
            "meters_gained": self.meters_gained,
            "sequence_count": self.sequence_count,
            "try_scored": self.try_scored,
            "decision_log": self.decision_log,
            "ball_positions": self.ball_positions,
            "meters_per_sequence": self.meters_per_sequence
        }

def run_single_simulation(_):
    model = RugbyModel()
    offensive_policy = OffensivePolicy()
    defensive_policy = DefensivePolicy(model)
    logger = SimulationLogger()
    simulation = RugbySimulation(model, offensive_policy, defensive_policy)
    simulation.run_simulation(logger, visualize=False)  # Disable visualization
    return logger.get_metrics()


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

"""
To run the code with the simulation visual remove all instances of "logger" within classes 
Uncomment the bellow main, 
Comment out all instances of logger, 
Comment out the main underneath it
Comment out the rugby evaluator class and the simulation logger

"""

 
# if __name__ == "__main__":
# #     Initialize the rugby model
#     model = RugbyModel()
#     offensive_policy = OffensivePolicy()
#     defensive_policy = DefensivePolicy(model)
#     visualizer = RugbyVisualizer(model)
#     simulation = RugbySimulation(model, offensive_policy, defensive_policy, visualizer)

#     # Visualize the simulation
#     simulation.run_simulation(visualize=True)
#     plt.show()

    
if __name__ == "__main__":
    num_simulations = 100000
    evaluator = RugbyEvaluator()
    print(f"Running {num_simulations} simulations on {cpu_count()} cores...")
    start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_single_simulation, range(num_simulations))
    for result in results:
        evaluator.log_simulation(result)

    end_time = time.time()
    print(f"Completed {num_simulations} simulations in {end_time - start_time:.2f} seconds.")
    evaluator.save_to_csv("simulation_results_100k.csv")
    evaluator.try_conversion_rate()
    evaluator.save_metrics_to_txt("simulation_metrics_100k.txt")
    evaluator.meters_gained_distribution()
    # evaluator.meters_gained_per_sequence() #struggle on 100k
    evaluator.decision_frequency()
    evaluator.ball_position_heatmap()
    evaluator.decision_success_rate()
    evaluator.transition_probability_matrix()