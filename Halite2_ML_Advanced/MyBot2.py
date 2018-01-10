import os

VERSION = 1

HM_ENT_FEATURES = 5
PCT_CHANGE_CHANCE = 30
DESIRED_SHIP_COUNT = 20

game = hlt.Game("Charles{}".format(VERSION))
logging.info("CharlesBot-{} Start".format(VERSION))

ship_plans = {}

if os.path.exists("c{}_input.vec").format(VERSION)):
    os.remove("c{}_input.vec".format(VERSION))

if os.path.exists("c{}_out.vec".format(VERSION)):
    os.remove("c{}_out.vec".format(VERSION))
