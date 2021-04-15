import sys
import os
from subsets import *

def main(directory):
    subset = top_50 # CHANGE IF NECESSARY
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:

            #  debugging
            #  print(filename)

            # Grab coin names
            coins, _ = filename.split('.')
            c1, c2 = coins.split('-')

            # Handle blacklist
            if c1 in blacklist or c2 in blacklist:
                os.remove(dirname + '/' + filename)
                continue

            # print target for now
            if (c1 in subset and c2 in stable) or (c2 in subset and c1 in stable):
                print(dirname + "/" + filename)
            else:
                to_remove = dirname + '/' + filename
                print("Would remove {} here!".format(to_remove))
                #  os.remove(to_remove)

print("Did you pass the directory as an arg?")
print(sys.argv)
#  pass the command line arg directory to the walker
main(sys.argv[1])
