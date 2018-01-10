import hlt
import logging
from collections import OrderedDict
import numpy as np
import random
import os

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
