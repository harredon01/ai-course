import random
#import numpy as np
import sys
import json
import time
from read import readInput
#from host2 import GO
from copy import deepcopy
from write import writeOutput
WIN_REWARD = 1
DRAW_REWARD = 0
LOSS_REWARD = -1

class HoovPlayer():
    def __init__(self, alpha=.7, gamma=.9, initial_value=0, side=None):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.side = side
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = {}
        self.history_states = []
        self.initial_value = initial_value
        self.type = 'random'
        #self.type_c = 1
    
    def move(self, go):
        #exploration_rate = 0.01 + \
    #(1 - 0.01) * np.exp(-0.001*100)
        #exploration_rate_threshold = random.uniform(0, 1)
        #if exploration_rate_threshold > exploration_rate:
        #    row, col = self.get_move(go,self.type_c)
        #else:
        #    row, col = self._select_best_move(go)

        move,is_savable, my_placements,whre = self.get_move(go,self.type_c)
        #go.visualize_board()
        #print(move,is_savable, my_placements)
        if move[0] == -1:
            return "PASS"
        if is_savable:
            self.history_states.append((go.encode_state(), move,my_placements))
        return move

    def Q(self, state):
        if state not in self.q_values:
            q_val = np.zeros((5, 5))
            q_val.fill(self.initial_value)
            self.q_values[state] = q_val
        return self.q_values[state]

    def get_possible_placements(self,go,piece_type):
        possible_placements = []
        for i in range(5):
            for j in range(5):
                if go.valid_place_check(i, j, piece_type, test_check = True):
                    possible_placements.append((i,j))
        return possible_placements

    def _select_random_move(self, board):
        possible_placements = self.get_possible_placements(board)
        if not possible_placements:
            return (-1,-1) #"PASS"
        else:
            return random.choice(possible_placements)

    def get_points(self,go):
        blacks = 0
        whites = 0
        blacks_l = 0
        whites_l = 0
        blacks_l2 = 0
        whites_l2 = 0
        blacks_a = 0
        whites_a = 0
        w_l = []
        b_l = []
        w_l2 = []
        b_l2 = []
        tot_with_allies_b = []
        tot_with_allies_w = []
        placements_b=[]
        placements_w=[]
        white_groups = []
        black_groups = []
        board = go.board
        #print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 1:
                    blacks +=1
                    pt = (i,j)
                    if pt not in tot_with_allies_b:
                        allies = go.ally_dfs(i,j)
                        len_allies = len(allies)
                        liberties_t = go.get_liberties(i,j,allies)
                        for it in liberties_t:
                            if it not in b_l:
                                #nbrs = go.detect_neighbor(it[0],it[1])
                                #for jt in nbrs:
                                #    if jt not in b_l2:
                                #        b_l2.append(jt)
                                #        if board[jt[0]][jt[1]] == 0:
                                #            blacks_l2+=1


                                b_l.append(it)
                                #blacks_l +=len_allies
                        cluster = {}
                        cluster['len_liberties']=len(liberties_t)
                        blacks_l +=(len_allies*cluster['len_liberties'])
                        cluster['allies'] = allies
                        cluster['liberties']= liberties_t
                        cluster['len_allies']=len_allies
                        blacks_a += len_allies
                        cluster['len_liberties']=len(liberties_t)
                        black_groups.append(cluster)
                        if len_allies>1:
                            for it in allies:
                                if it not in tot_with_allies_b:
                                    tot_with_allies_b.append(it)
                elif board[i][j] == 2:
                    whites +=1
                    pt = (i,j)
                    if pt not in tot_with_allies_w:
                        allies = go.ally_dfs(i,j)
                        len_allies = len(allies)
                        liberties_t = go.get_liberties(i,j,allies)
                        for it in liberties_t:
                            if it not in w_l:
                                #nbrs = go.detect_neighbor(it[0],it[1])
                                #for jt in nbrs:
                                #    if jt not in w_l2:
                                #        w_l2.append(jt)
                                #        if board[jt[0]][jt[1]] == 0:
                                #            whites_l2+=1
                                w_l.append(it)
                                #whites_l +=len_allies
                        
                        cluster = {}
                        len_allies = len(allies)
                        cluster['len_liberties']=len(liberties_t)
                        whites_l +=(len_allies*cluster['len_liberties'])
                        cluster['allies'] = allies
                        cluster['liberties']= liberties_t
                        cluster['len_allies']=len_allies
                        whites_a += len_allies
                        cluster['len_liberties']=len(liberties_t)
                        white_groups.append(cluster)
                        if len_allies>1:
                            for it in allies:
                                if it not in tot_with_allies_w:
                                    tot_with_allies_w.append(it)
                elif board[i][j]==0:
                    if go.valid_place_check(i,j,1):
                        placements_b.append((i,j))
                    if go.valid_place_check(i,j,2):
                        placements_w.append((i,j))
        #self.visualize_board(go.board)
        return white_groups,black_groups,whites,blacks,placements_w,placements_b,whites_l,blacks_l,whites_a,blacks_a,whites_l2,blacks_l2

    def get_qlearning_move(self, go):
        state = go.encode_state()
        info_exists = False
        original = state
        ress = self.try_alternates(state)
        if ress:
            info_exists = True
            print("Alternatives worked",ress)
            state = ress
        else:
            return None,None
            

        q_values = self.Q2(state,[])
        res = sorted(q_values.items(), key=lambda item: item[1], reverse=True)
        if len(res)==0:
            return -1,-1
        da_obj = res[0]
        x = da_obj[0]
        x = self.reverse_state(state,x,original)
        return int(x[0]),int(x[1])
            
        

    def _select_best_move2(self, board):
        state = board.encode_state()
        info_exists = False
        ress = self.try_alternates(state)
        if ress:
            info_exists = True
            print("Alternatives worked",ress)
            state = ress

        q_values = self.Q(state)
        row, col = -1, -1
        curr_max = -np.inf
        while True:
            i, j,max = self._find_max(q_values)
            if max <= 0:
                print("Resorting to random",max)
                return self._select_random_move(board)
            row, col = i, j

            if board.valid_place_check(i, j,self.type_c):
                print("q workeddddddddddddddddddddddddddddddddddddd")
                return i, j
            else:
                q_values[i][j] = -1.0

    def _find_max(self, q_values):
        curr_max = -np.inf
        row, col = 0, 0
        for i in range(0, 5):
            for j in range(0, 5):
                if q_values[i][j] > curr_max:
                    curr_max = q_values[i][j]
                    row, col = i, j
        return row, col,curr_max
    def save_q(self):
        json_object = json.dumps(self.q_values, separators=(',', ':'))
        with open("q_values.json", "w") as outfile:
            outfile.write(json_object)
        print(self.q_values)
        print("DONEEE")

    def load_q(self):
        with open('q_values.json', 'r') as openfile:
            obj_q = json.load(openfile)
        self.q_values = obj_q
        print("loaded q")

    def find_holder(self,board):
        print("Find holder")
        for i in range(5):
            for j in range(5):
                if board[i][j]== 3:
                    return (i,j)

    def try_alternates(self,encoded,move):
        if self.type_c == 2:
            encoded = encoded.replace("1","3")
            encoded = encoded.replace("2","1")
            encoded = encoded.replace("3","2")
        if encoded in self.q_values:
            return encoded,move
        rez = self.decode_state(encoded)
        if move[0]>=0:
            rez[move[0]][move[1]]=3
        rez1 = [[rez[j][i] for j in range(5)] for i in range(5)]
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez)
        rez.reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez)
        rez1.reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez1)
        for it in range(5):
            rez[it].reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez)
        for it in range(5):
            rez1[it].reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez1)
        rez1.reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez1)

        rez.reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,self.find_holder(rez)
        return None,move

    def try_alternates2(self,encoded,type_c):
        if type_c == 2:
            encoded = encoded.replace("1","3")
            encoded = encoded.replace("2","1")
            encoded = encoded.replace("3","2")
        if encoded in self.q_values:
            return encoded
        rez = self.decode_state(encoded)
        rez1 = [[rez[j][i] for j in range(5)] for i in range(5)]
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1,
        rez.reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1
        rez1.reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1
        for it in range(5):
            rez[it].reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1
        for it in range(5):
            rez1[it].reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1
        rez1.reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1

        rez.reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1 in self.q_values:
            return state1
        return None

    def reverse_state(self,encoded,move,target,type_c):
        if type_c == 2:
            target = target.replace("1","3")
            target = target.replace("2","1")
            target = target.replace("3","2")
        if encoded == target:
            return move
        rez = self.decode_state(encoded)
        rez[move[0]][move[1]]=3
        rez1 = [[rez[j][i] for j in range(5)] for i in range(5)]
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            return self.find_holder(rez1)
        rez.reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            #self.visualize_board(rez)
            return self.find_holder(rez)
        rez1.reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            return self.find_holder(rez1)
        for it in range(5):
            rez[it].reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            return self.find_holder(rez)
        for it in range(5):
            rez1[it].reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            return self.find_holder(rez1)
        rez1.reverse()
        state1 = ''.join([str(rez1[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            return self.find_holder(rez1)

        rez.reverse()
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        if state1.replace("3","0") == target:
            return self.find_holder(rez)
        return None
    def decode_state(self,encoded):
        board = [[0 for x in range(5)] for y in range(5)]

        for it in range(len(encoded)):
            i = 0
            j = 0
            if it < 5:
                j = it
            elif it >= 5 and it < 10:
                i = 1
                j = it - 5
            elif it >= 10 and it < 15:
                i = 2
                j = it - 10
            elif it >= 15 and it < 20:
                i = 3
                j = it - 15
            elif it >= 20 and it < 25:
                i = 4
                j = it - 20
            board[i][j] = int(encoded[it])
        return board
    
    def visualize_board(self,board):
        '''
        Visualize the board.
        :return: None
        '''
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

    def try_sim(self):
        rez = [[1,2,1,2,1],[1,1,0,0,0],[1,1,1,1,1],[2,2,2,2,2],[0,0,0,2,2]]
        state1 = ''.join([str(rez[i][j]) for i in range(5) for j in range(5)])
        print(state1)
        self.try_alternates(state1)

    def learn(self, game_result,test_board):
        """ when games ended, this method will be called to update the qvalues
        """
        if game_result == 0:
            reward = DRAW_REWARD
        elif game_result == self.type_c:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        past = None
        is_found = False
        for hist in self.history_states:
            state, move,placements = hist
            #self.visualize_board(self.decode_state(state))
            #print("pres",move)
            prev_state = state
            ress,move = self.try_alternates(state,move)
            if ress:
                state = ress
            q = self.Q2(state,placements)
            res = sorted(q.items(), key=lambda item: item[1], reverse=True)
            for it in res:
                da_obj = res[0]
                max_q_value = da_obj[1]
                break
            key = str(move[0])+str(move[1])
            try:
                q[key] = round(q[key] * (1 - self.alpha) + self.alpha *( reward + self.gamma * max_q_value),4)
            except:
                print("Exception")
                self.visualize_board(self.decode_state(state))
                self.visualize_board(self.decode_state(prev_state))
                print(q)
                print(key)
                print(placements)
                print("pres",state)
                print("past",past)
                #exit()
            past = state
            #print("Reward",reward,"qval",q[move[0]][move[1]])
            
        #print("Learn result")
        
        self.history_states = []

    def get_max(self,q_val):
        max = -1000
        win = ""
        for it in q_val:
            if q_val[it]> max:
                max = q_val[it]
                win = it
        return max,win


    def Q2(self, state,possible_placements):
        if state not in self.q_values:
            q_val = {}
            for it in range(len(possible_placements)):
                key = str(possible_placements[it][0])+str(possible_placements[it][1])
                q_val[key]=self.initial_value
            self.q_values[state] = q_val
        return self.q_values[state]

    def learn2(self, game_result):
        """ when games ended, this method will be called to update the qvalues
        """
        if game_result == 0:
            reward = DRAW_REWARD
        elif game_result == self.type_c:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        self.history_states.reverse()
        max_q_value = -1.0
        is_found = False
        for hist in self.history_states:
            state, move = hist
            ress = self.try_alternates(state)
            if ress:
                state = ress
            q = self.Q(state)
            max_q_value = np.max(q)
            q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha *( reward + self.gamma * max_q_value)
            #print("Reward",reward,"qval",q[move[0]][move[1]])
            
        #print("Learn result")
        
        self.history_states = []
    
    def set_type(self,type):
        self.type_c = type

    def get_successors(self,go,piece_type):
        placements = self.get_possible_placements(go,piece_type)
        children = []
        scores = []
        hass = {}


        goods = {}
        state = go.encode_state()
        ress = self.try_alternates2(state,piece_type)
        if False:
            q_values = self.Q2(ress,placements)
            for i in q_values:
                x = None
                if q_values[i] < 0 or q_values[i] > 0:
                    x = (int(i[0]),int(i[1]))
                    #self.visualize_board(self.decode_state(state))
                    #self.visualize_board(self.decode_state(ress))
                    x = self.reverse_state(ress,x,state,piece_type)
                    goods[x] = q_values[i] 



        factor = 10
        counter = 0
        for it in placements:
            cand = {}
            cp = go.copy_board()

            additional = 0
            if it in goods:
                additional = goods[it]*(-10)
            died_pieces,en_pieces,my_pieces,en_libs = cp.make_move(it,piece_type)
            cp.last_move = it
            cp.dieds = died_pieces
            cand['board']=counter
            cand['score']=(en_pieces-len(died_pieces)-my_pieces)*4 + en_libs + additional
            hass[counter]=cp
            children.append(cp)
            scores.append(cand)
            counter +=1
        if len(children)<=factor:
            return children
        else:
            newlist = sorted(scores, key=lambda d: d['score'], reverse=False)
            final = []
            for i in range(factor):
                if(hass[newlist[i]['board']]):
                    final.append(hass[newlist[i]['board']])
            return final


    def alpha_beta_search(self,go,piece_type):
        best_val = -100000
        beta = 100000
        max_debth = 3
        #print("Root")
        #self.visualize_board(go.board)
        successors = self.get_successors(go,piece_type)
        #print(len(successors))
        best_state = None
        #print(self.get_total_points(go,piece_type))
        for state in successors:
            #state = go.copy_board()
            #state.update_board(obj['board'])
            value = self.min_value(state, best_val, beta,max_debth,3 - piece_type,piece_type)
            #print("value")
            #self.visualize_board(state.board)
            #print("first",self.get_total_points(state,piece_type))
            #print("final",value)
            #print(state.last_move)
            if value > best_val:
                best_val = value
                best_state = state
        #print("AlphaBeta:  Utility Value of Root Node: = " + str(best_val))
        #print("AlphaBeta:  Best State is: ")
        #board = best_state.board
        #self.visualize_board(board)
        #print(self.get_total_points(go,piece_type))
        if best_state:
            #print("a",best_state.board)
            return best_state.last_move
        else:
            return (-1,-1)

    def is_terminal(self,board):
        empties = 0
        for i in range(5):
            for j in range(5):
                if board[i][j]==0:
                    return False
        return True

    def max_value(self, go, alpha, beta,max_debth,piece_type,my_piece):
        max_debth -=1
        term = self.is_terminal(go.board)
        if term or max_debth == 0:
            if term:
                winner = go.judge_winner()
                if winner == my_piece:
                    return 100
                else:
                    return -100

            points = self.get_total_points(go,my_piece)
            #print("max leaf points",points)
            #self.visualize_board(go.board)
            return points
        infinity = 100000
        value = -100000

        successors = self.get_successors(go,piece_type)
        for state in successors:
            #state = go.copy_board()
            #state.update_board(obj['board'])
            value = max(value, self.min_value(state, alpha, beta,max_debth,3 - piece_type,my_piece))
            #print("Max",value)
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, go, alpha, beta,max_debth,piece_type,my_piece):
        max_debth -=1
        term = self.is_terminal(go.board)
        if term or max_debth == 0:
            if term:
                winner = go.judge_winner()
                if winner == my_piece:
                    return 100
                else:
                    return -100
            points = self.get_total_points(go,my_piece)
            #print("min leaf points",points)
            #self.visualize_board(go.board)
            return points
            print("mindieds",len(go.dieds))
            return len(go.dieds)
        infinity = 100000
        value = infinity

        successors = self.get_successors(go,piece_type)
        for state in successors:
            #state = go.copy_board()
            #state.update_board(obj['board'])
            value = min(value, self.max_value(state, alpha, beta,max_debth,3 - piece_type,my_piece))
            #print("Min",value)
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''        
        
        self.type_c = piece_type
        row, col = self._select_best_move(go)
        
        if row == -1:
            return "PASS"
        return (row,col)
    def get_total_points(self,go,piece_type):
        white_groups,black_groups,whites,blacks,placements_w,placements_b,whites_l,blacks_l,whites_a,blacks_a,whites_l2,blacks_l2 = self.get_points(go)
        if len(white_groups)>0 and len(black_groups)>0:
            whites_a = whites_a/len(white_groups)
            blacks_a = blacks_a / len(black_groups)
        else:
            whites_a = 0
            blacks_a = 0
        if piece_type == 1:
            #return (blacks*3+blacks_l + blacks_l2 + blacks_a) - ((whites+2.5)*3 + whites_l + whites_l2+whites_a)
            return (blacks*4 +blacks_l + blacks_l2)-((whites+2.5)*5+whites_l+whites_l2)
        else:
            #return ((whites+2.5)*3 + whites_l + whites_l2 + whites_a) - (blacks*3+blacks_l + blacks_l2+blacks_a)
            return ((whites+2.5)*5+whites_l+whites_l2)-(blacks*5 +blacks_l + blacks_l2)

    def get_move(self,go,piece_type):
        
        my_placements = []
        my_groups = []
        en_libs = []
        white_groups,black_groups,whites,blacks,placements_w,placements_b,whites_l,blacks_l,whites_a,blacks_a,whites_l2,blacks_l2 = self.get_points(go)
        total  = blacks + whites
        if piece_type == 1:
            my_placements = placements_b
            my_groups = black_groups
            en_libs = whites_l
        else:
            my_placements = placements_w
            my_groups = white_groups
            en_libs = blacks_l
        #newlist = sorted(en_libs, key=lambda d: d['len_liberties'], reverse=False)
        #for it in newlist:
            #libs = it['liberties']
            #print("Move",libs[0])
            #if go.valid_place_check(libs[0][0], libs[0][1], piece_type, test_check = True):
                #return libs[0],True,my_placements,0
        if not my_placements:
            return (-1,-1),True,my_placements,0
        if total < 3:
            p = (2,2)
            if p in my_placements:
                return p,False,my_placements,7
            for i in range (3):
                for j in range (3):
                    p = (i+1,j+1)
                    if p in my_placements:
                        return p,False,my_placements,1
        elif False:
            #print(my_groups)
            newlist = sorted(my_groups, key=lambda d: d['len_allies'], reverse=True)
            cands = [] 
            max_group = 0
            board = go.board
            win_lib = {}
            for it in newlist:
                 libs = it['liberties']
                 for i in libs:
                    neib = go.detect_neighbor(i[0],i[1])
                    empts = 0
                    for j in neib:
                        if board[j[0]][j[1]]==0:
                            empts+=1
                    cont = {}
                    cont['total'] = it['len_allies'] + empts
                    cont['pos']=i
                    cands.append(cont)
            cands = sorted(cands, key=lambda d: d['total'], reverse=True)
            for it in cands:
                if it['pos'] in my_placements:
                    return it['pos'],True,my_placements,2
        else:
            #print("alpha_beta_search")
            return self.alpha_beta_search(go,piece_type),True,my_placements,3
        
            






if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    startA = time.time()
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    go.type_c = piece_type
    player = HoovPlayer()
    #curr_state = ''.join([str(board[i][j]) for i in range(5) for j in range(5)])
    #player.visualize_board(previous_board)
    #player.visualize_board(board)
    #print("type",go.type_c)
    player.load_q()
    action = player.move(go)
    print("Complete: "+ str((time.time()) - startA))
    #print("move",action)
    writeOutput(action)