import sys
import random
import timeit
import math
import argparse
from collections import Counter
from copy import deepcopy
from my_player3 import HoovPlayer
#from hoov_player3 import HoovPlayer
from hoov_player2 import HoovPlayer2
import numpy as np
from read import *
from write import writeNextInput

class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.last_move = (-1,-1) # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 # Trace the number of moves
        self.max_move = n * n - 1 # The max movement of a Go game
        self.komi = n/2 # Komi rule
        self.verbose = False # Verbose only when there is a manual player

    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        en_pieces = 0
        en_libs = 0
        my_pieces = 0
        died_pieces = []
        my_piece = 3 - piece_type

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    en_pieces +=1
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
                    else:
                        en_libs+=1
                elif board[i][j]== my_piece:
                    my_pieces +=1

        return died_pieces,en_pieces,my_pieces,en_libs

    def remove_died_pieces2(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces,en_pieces,my_pieces,en_libs = self.find_died_pieces(piece_type)
        if not died_pieces: return [],en_pieces,my_pieces,en_libs 
        self.remove_certain_pieces(died_pieces)
        return died_pieces,en_pieces,my_pieces,en_libs

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces,en_pieces,my_pieces,en_libs = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True
        
    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

    def visualize_board(self):
        '''
        Visualize the board.
        :return: None
        '''
        board = self.board
        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt

    def reverse_board(self,board):
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 1:
                    board[i][j] = 2
                elif board[i][j] == 2:
                    board[i][j] = 1
        return board


    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''        

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi: return 1
        elif cnt_1 < cnt_2 + self.komi: return 2
        else: return 0

    def make_move(self,action,piece_type):
        # If invalid input, continue the loop. Else it places a chess on the board.
        self.place_chess(action[0], action[1], piece_type)
        #if True:
            #self.visualize_board() 


        return self.remove_died_pieces2(3 - piece_type) # Remove the dead pieces of opponent
        
    def play(self, player1, player2, episode):
        '''
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        self.init_board(self.size)
        exploration_rate = 0.01 + \
        (1 - 0.01) * np.exp(-0.001*episode)
        #print(exploration_rate)

        # Print input hints and error message if there is a manual player
        
        verbose = False
        # Game starts!
        while 1:
            piece_type = 1

            # Judge if the game should end
            if self.game_end(piece_type):       
                result = self.judge_winner()
                if True:
                    print('Game ended.')
                    if result == 0: 
                        print('The game is a tie.')
                    else: 
                        print('The winner is {}'.format('X' if result == 1 else 'O'))
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")

            # Game continues
            action = player1.move( self)

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board() 
                    continue

                self.died_pieces = self.remove_died_pieces(3 - piece_type) # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board() # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move # Players take turn

            piece_type = 2

            # Judge if the game should end
            if self.game_end(piece_type):       
                result = self.judge_winner()
                if verbose:
                    print('Game ended.')
                    if result == 0: 
                        print('The game is a tie.')
                    else: 
                        print('The winner is {}'.format('X' if result == 1 else 'O'))
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")

            # Game continues
            action = player2.move( self)
            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board() 
                    continue

                self.died_pieces = self.remove_died_pieces(3 - piece_type) # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board() # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move # Players take turn
    def get_liberties(self, i, j,ally_members):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        
        liberties = []
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    pt = (piece[0],piece[1])
                    if pt not in liberties:
                        liberties.append(pt)
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return liberties
    def get_points(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board
        blacks = 0
        whites = 0
        biggest_white = 0
        biggest_black = 0
        tot_with_allies_b = []
        tot_with_allies_w = []
        liberties_b = []
        liberties_w = []

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 1:
                    blacks +=1
                    pt = (i,j)
                    if pt not in tot_with_allies_b:
                        allies = self.ally_dfs(i,j)
                        liberties_t = self.get_liberties(i,j,allies)
                        for item in liberties_t:
                            if item not in liberties_b:
                                liberties_b.append(item)
                        len_allies = len(allies)
                        if len_allies>1:
                            if len_allies > biggest_black:
                                biggest_black = len_allies
                            for it in allies:
                                if it not in tot_with_allies_b:
                                    tot_with_allies_b.append(it)
                elif board[i][j] == 1:
                    print('X', end=' ')
            print()
        print('-' * len(board) * 2)

    
    def encode_state(self):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(self.board[i][j]) for i in range(self.size) for j in range(self.size)])

    


def judge(n_move, verbose=False):

    N = 5
   
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.verbose = verbose
    go.set_board(piece_type, previous_board, board)
    go.n_move = n_move
    try:
        action, x, y = readOutput()
    except:
        print("output.txt not found or invalid format")
        sys.exit(3-piece_type)

    if action == "MOVE":
        if not go.place_chess(x, y, piece_type):
            print('Game end.')
            print('The winner is {}'.format('X' if 3 - piece_type == 1 else 'O'))
            sys.exit(3 - piece_type)

        go.died_pieces = go.remove_died_pieces(3 - piece_type)

    if verbose:
        go.visualize_board()
        print()

    if go.game_end(piece_type, action):       
        result = go.judge_winner()
        if verbose:
            print('Game end.')
            if result == 0: 
                print('The game is a tie.')
            else: 
                print('The winner is {}'.format('X' if result == 1 else 'O'))
        sys.exit(result)

    piece_type = 2 if piece_type == 1 else 1

    if action == "PASS":
        go.previous_board = go.board
    writeNextInput(piece_type, go.previous_board, go.board)

    sys.exit(0)

def battle(board, player1, player2, iter, learn=False, show_result=True):
    p1_stats = [0, 0, 0] # draw, win, lose
    try:
        player1.load_q()
        player2.load_q()
    except:
        print("An exception occurred")
    # Copy the board for testing
    test_go = board.copy_board()
    for i in range(0, iter):
        print("Playing game: ",i)
        result = board.play(player1, player2, i)
        p1_stats[result] += 1
        player1.learn(result,test_go)
        player2.learn(result,test_go)
        print(i)
    player1.save_q()
    player2.save_q()
    

    p1_stats = [round(x / iter * 100.0, 1) for x in p1_stats]
    if True:
        print('_' * 60)
        print('{:>15}(X) | Wins:{}% Draws:{}% Losses:{}%'.format(player1.__class__.__name__, p1_stats[1], p1_stats[0], p1_stats[2]).center(50))
        print('{:>15}(O) | Wins:{}% Draws:{}% Losses:{}%'.format(player2.__class__.__name__, p1_stats[2], p1_stats[0], p1_stats[1]).center(50))
        print('_' * 60)
        print()

    return p1_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ranPlayer = HoovPlayer2()
    hoovPlayer = HoovPlayer()
    #ranPlayer2 = RandomPlayer()
    
    #hoovPlayer.try_sim()
    #exit()
    N = 5
    parser.add_argument("--move", "-m", type=int, help="number of total moves", default=0)
    parser.add_argument("--verbose", "-v", type=bool, help="print board", default=False)
    args = parser.parse_args()
    go = GO(N)
    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  10, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 1")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 1")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 2")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 3")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 4")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 5")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 6")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 7")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 8")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 9")

    hoovPlayer.set_type(1)
    ranPlayer.set_type(2)
    battle(go,hoovPlayer,ranPlayer,  100, learn=True, show_result=False)
    hoovPlayer.set_type(2)
    ranPlayer.set_type(1)
    battle(go,ranPlayer,hoovPlayer,  100, learn=True, show_result=False)
    print(" pair 10")

    # judge(args.move, args.verbose)
        
        
