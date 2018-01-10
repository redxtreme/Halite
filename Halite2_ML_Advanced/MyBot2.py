import hlt
import logging
from collections import OrderedDict
import numpy as np
import random
import os

# Reference video: https://www.youtube.com/watch?v=KPBRWF7ALPQ&t=549s
# Reference site: https://pythonprogramming.net/deep-learning-halite-ii-artificial-intelligence-competition/
VERSION = 1

HM_ENT_FEATURES = 5 # Features per entity to track
PCT_CHANGE_CHANCE = 30 # Chance of randomness in change of plan
DESIRED_SHIP_COUNT = 20 # 20-50, but 50 times out some times. 2 sec max

game = hlt.Game("Charles{}".format(VERSION))
logging.info("CharlesBot-{} Start".format(VERSION))

ship_plans = {}

# If the file exists, delete it
if os.path.exists("c{}_input.vec").format(VERSION)):
    os.remove("c{}_input.vec".format(VERSION))

if os.path.exists("c{}_out.vec".format(VERSION)):
    os.remove("c{}_out.vec".format(VERSION))

# Find dictionary key by the value, key is the distance
def key_by_value(dictionary, value):
    for k,v in dictionary.items():
        if v[0] == value:
            return k
    return -99

# Coalesce data if we dont have HM_ENT_FEATURES number of features (ships/planets)
def fix_data(data):
    new_list = []
    last_known_idx = 0
    for i in range(HM_ENT_FEATURES):
        try:
            if i < len(data):
                last_known_idx = i
            new_list.append(data[last_known_idx])
        except:
            new_list.append(0)
    return new_list

while True:
    game_map = game.update_map()
    command_queue = []

    team_ships = game_map.get_me().all_ships()
    all_ships = game_map._all_ships()
    enemy_ships = [ships for ship in game_map._all_ships() if ship not in team_ships]

    my_ship_count = len(team_ships)
    enemy_ship_count = len(enemy_ships)
    all_ship_count = len(all_ships)

    my_id = game_map.get_me().id

    empty_planet_sizes = {}
    our_planet_sizes = {}
    enemy_planet_sizes = {}

    # For every planet in the map
    for p in game_map.all_planets():
        radius = p.radius

        # If not owned by anyone
        if not p.is_owned():
            empty_planet_sizes[radius] = p

        # else if we are the owner
        elif p.owner.id == game_map.get_me().id:
            our_planet_sizes[radius] = p

        # if owned but not by us
        elif p.owner.id != game_map.get_me().id:
            enemy_planet_sizes[radius] = p

    hm_our_planets = len(our_planet_sizes)
    hm_empty_planets = len(empty_planet_sizes)
    hm_enemy_planets = len(enemy_planet_sizes)

    # Use the sizes to sort them, largests to smallest
    our_planet_keys = sorted([k for k in empty_planet_sizes])[::-1] # Reverse order
    empty_planet_keys = sorted([k for k in empty_planet_sizes])[::-1] # Reverse order
    enemy_planet_keys = sorted([k for k in empty_planet_sizes])[::-1] # Reverse order

    for ship in game_map.get_me().all_ships():
        try:

            # If ship is docked, skip it.
            # Might want to leave this out. It makes the ships aggressive
            if ship.docking_status != ship.DockingStatus.UNDOCKED:
                # Skip this ship
                continue

                shipid = ship.id
                change = False
                if random.randint(1, 100) <= PCT_CHANGE_CHANCE:
                    change = True

            # Grabs enities by distance
            entities_by_distance = game_map.nearby_entities_by_distance(ship)
            # Sort that by the key, closest to furthest
            entities_by_distance = OrderedDict(sorted(entities_by_distance.items(), key=lambda t: t[0]))

            # Get the entity elements (those with no owner)
            closest_empty_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not entities_by_distance[distance][0].is_owned()]
            # Raw distance used for training
            closest_empty_planet_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and not entities_by_distance[distance][0].is_owned()]

            # Same for our own
            closest_my_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0].is_owned() and (entities_by_distance[distance][0].owner.id == game_map.get_me().id)]
            closest_my_planets_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0].is_owned() and (entities_by_distance[distance][0].owner.id == game_map.get_me().id)]

            # Same for enemy
            closest_enemy_planets = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0] not in closest_my_planets and entities_by_distance[distance][0] not in closest_empty_planets]
            closest_enemy_planets_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Planet) and entities_by_distance[distance][0] not in closest_my_planets and entities_by_distance[distance][0] not in closest_empty_planets]

            # Same for closest team ships
            closest_team_ships = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and entities_by_distance[distance][0] in team_ships]
            closest_team_ships_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and entities_by_distance[distance][0] in team_ships]

            # Same for closest enemy ships
            closest_enemy_ships = [entities_by_distance[distance][0] for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and entities_by_distance[distance][0] not in team_ships]
            closest_enemy_ships_distances = [distance for distance in entities_by_distance if isinstance(entities_by_distance[distance][0], hlt.entity.Ship) and entities_by_distance[distance][0] not in team_ships]

            largest_empty_planet_distances = []
            largest_our_planet_distances = []
            largest_enemy_planet_distances = []

            for i in range(HM_ENT_FEATURES):
                try: largest_empty_planet_distances.append(key_by_value(entities_by_distance, empty_planet_sizes[empty_planet_keys[i]]))
                except:largest_empty_planet_distances.append(-99) # Default to dist -99

                try: largest_our_planet_distances.append(key_by_value(entities_by_distance, our_planet_sizes[our_planet_keys[i]]))
                except:largest_our_planet_distances.append(-99) # Default to dist -99

                try: largest_enemy_planet_distances.append(key_by_value(entities_by_distance, enemy_planet_sizes[enemy_planet_keys[i]]))
                except:largest_enemy_planet_distances.append(-99) # Default to dist -99

            entity_lists = [fix_data(closest_empty_planet_distances),
                            fix_data(closest_my_planets_distances),
                            fix_data(closest_enemy_planets_distances),
                            fix_data(closest_team_ships_distances),
                            fix_data(closest_enemy_ships_distances),
                            fix_data(empty_planet_keys),
                            fix_data(our_planet_keys),
                            fix_data(enemy_planet_keys),
                            fix_data(largest_empty_planet_distances),
                            fix_data(largest_our_planet_distances),
                            fix_data(largest_enemy_planet_distances

            input_vector = []

            # Itterate over the list and append them to the input vector
            for i in entity_lists:
                # 5 closest features
                for ii in i[:HM_ENT_FEATURES]:
                    input_vector.append(ii)

            input_vector += [my_ship_count,
                            enemy_ship_count,
                            hm_our_planets,
                            hm_empty_planets,
                            hm_enemy_planets]

            # Calculate the output vector as random (for now)
            if my_ship_count > DESIRED_SHIP_COUNT:
                ''' Attack: [1, 0, 0] '''
                output_vector = 3 * [0] # [0, 0, 0]
                output_vector[0] = 1 # [1, 0, 0]
                ship_plans[ship.id] = output_vector

            # Ship doesn't have a plan or change is True
            elif change or ship.id not in ship_plans:
                # TODO this changes if using a NN. NN will chose output vector
                ''' Pick a new "plan" '''
                output_vector = 3*[0]
                output_vector[random.randint(0,2)] = 1
                ship_plans[ship.id] = output_vector

            else:
                ''' Continue to execute existing plans '''
                output_vector = ship_plans[ship.id]

            try:

                # ATTACK ENEMY SHIP #
                if np.argmax(output_vector) == 0: #[1,0,0]
                    '''
                    type: 0
                    Find closest enemy ship, and attack!
                    '''
                    if not isinstance(closest_enemy_ships[0], int):
                        navigate_command = ship.navigate(
                                    ship.closest_point_to(closest_enemy_ships[0]),
                                    game_map,
                                    speed=int(hlt.constants.MAX_SPEED),
                                    ignore_ships=False)

                        if navigate_command:
                            command_queue.append(navigate_command)



                # MINE ONE OF OUR PLANETS #
                elif np.argmax(output_vector) == 1:
                    '''
                    type: 1
                    Mine closest already-owned planet
                    '''
                    if not isinstance(closest_my_planets[0], int):
                        target =  closest_my_planets[0]

                        if len(target._docked_ship_ids) < target.num_docking_spots:
                            if ship.can_dock(target):
                                command_queue.append(ship.dock(target))
                            else:
                                navigate_command = ship.navigate(
                                            ship.closest_point_to(target),
                                            game_map,
                                            speed=int(hlt.constants.MAX_SPEED),
                                            ignore_ships=False)

                                if navigate_command:
                                    command_queue.append(navigate_command)
                        else:
                            #attack!
                            if not isinstance(closest_enemy_ships[0], int):
                                navigate_command = ship.navigate(
                                            ship.closest_point_to(closest_enemy_ships[0]),
                                            game_map,
                                            speed=int(hlt.constants.MAX_SPEED),
                                            ignore_ships=False)

                                if navigate_command:
                                    command_queue.append(navigate_command)



                    elif not isinstance(closest_empty_planets[0], int):
                        target =  closest_empty_planets[0]
                        if ship.can_dock(target):
                            command_queue.append(ship.dock(target))
                        else:
                            navigate_command = ship.navigate(
                                        ship.closest_point_to(target),
                                        game_map,
                                        speed=int(hlt.constants.MAX_SPEED),
                                        ignore_ships=False)

                            if navigate_command:
                                command_queue.append(navigate_command)

                    #attack!
                    elif not isinstance(closest_enemy_ships[0], int):
                        navigate_command = ship.navigate(
                                    ship.closest_point_to(closest_enemy_ships[0]),
                                    game_map,
                                    speed=int(hlt.constants.MAX_SPEED),
                                    ignore_ships=False)

                        if navigate_command:
                            command_queue.append(navigate_command)




                # FIND AND MINE AN EMPTY PLANET #
                elif np.argmax(output_vector) == 2: # [0, 0, 1]
                    '''
                    type: 2
                    Mine an empty planet.
                    '''
                    if not isinstance(closest_empty_planets[0], int):
                        target =  closest_empty_planets[0]

                        if ship.can_dock(target):
                            command_queue.append(ship.dock(target))
                        else:
                            navigate_command = ship.navigate(
                                        ship.closest_point_to(target),
                                        game_map,
                                        speed=int(hlt.constants.MAX_SPEED),
                                        ignore_ships=False)

                            if navigate_command:
                                command_queue.append(navigate_command)

                    else:
                        #attack!
                        if not isinstance(closest_enemy_ships[0], int):
                            navigate_command = ship.navigate(
                                        ship.closest_point_to(closest_enemy_ships[0]),
                                        game_map,
                                        speed=int(hlt.constants.MAX_SPEED),
                                        ignore_ships=False)

                            if navigate_command:
                                command_queue.append(navigate_command)




            except Exception as e:
                logging.info(str(e))

            with open("c{}_input.vec".format(VERSION),"a") as f:
                f.write(str( [round(item,3) for item in input_vector] ))
                f.write('\n')

            with open("c{}_out.vec".format(VERSION),"a") as f:
                f.write(str(output_vector))
                f.write('\n')

        except Exception as e:
            logging.info(str(e))

    game.send_command_queue(command_queue)
    # TURN END
# GAME END
