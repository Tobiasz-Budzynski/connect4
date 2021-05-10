import numpy as np
from enum import Enum
from typing import Optional
import math

four = np.ones((1, 4))
four_connected = [four, four.T, np.diag(four.flatten()), np.fliplr(np.diag(four.reshape(4)))]
print(four_connected)

class Solver:

    def demo(self, a, b, c):
        d = b ** 2 - 4 * a * c
        if d > 0:
            disc = math.sqrt(d)
            root1 = (-b + disc) / (2 * a)
            root2 = (-b - disc) / (2 * a)

            return root1, root2
        elif d == 0:
            return -b / (2 * a)
        else:
            return "This equation has no roots"



if __name__ == '__main__':
    solver = Solver()

    while True:
        a = int(input("a: "))
        b = int(input("b: "))
        c = int(input("c: "))
        result = solver.demo(a, b, c)
        print(result)


#    freedom_j = GameDimensions.LENGTH.value - GameDimensions.CONNECT.value + 1
#    freedom_i = GameDimensions.HIGHT.value - GameDimensions.CONNECT.value + 1

# for i in np.arange(GameDimensions.HIGHT.value):
#     for j in np.arange(freedom_j):
board = initialize_game_state()
i = 0
j = 0

board[0, 0:GameDimensions.CONNECT.value] = 1
connect_position = (board == 1)
# todo: use np.roll to get all winning positions as matrices for indexing (checking the index).
