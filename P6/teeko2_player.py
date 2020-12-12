import random
import time
import copy
class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        
    
        
    def make_move(self, state): 
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = True   # TODO: detect drop phase
        num_chess = 0
        for i in range(5):
            for j in range(5):
                if not state[i][j] == ' ':
                    num_chess += 1
        if num_chess >= 8:
            drop_phase = False
            
         # first drop -- try it best to drop to the enter
        if sum(row.count(self.my_piece) for row in state) == 0:
            if state[2][2]==' ':
                r = 2
                c = 2
            elif not state[2][2] == ' ':
                r = 2
                c = 3
            first = []
            first.append((r,c))
            return first
        
        next_state = self.max_value(state, 3,'AI')[1]
        # drop phase
        if drop_phase:
            drop = []
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ' and next_state[row][col] == self.my_piece:
                        drop.append((row, col))
            return drop
        # move phase
        if not drop_phase:
            next_move=[]
            for row in range(5):
                for col in range(5):
                    if state[row][col]==self.my_piece and next_state[row][col]==' ':#it is this chess moved
                        next_move.append((row,col))
                    if state[row][col] == ' ' and next_state[row][col] == self.my_piece:#it is this position moved by a chess
                        next_move.insert(0,(row,col))
            return next_move


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and diamond wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1
        
        # check \ diagonal wins
        for i in range(2):
            if state[i][i] != ' ' and state[i][i] == state[i+1][i+1] == state[i+2][i+2] == state[i+3][i+3]:
                return 1 if state[i][col]==self.my_piece else -1
              
        # check / diagonal wins
        for i in range(2):
          if state[4-i][i] != ' ' and state[4-i][i] == state[4-i-1][i+1] == state[4-i-2][i+2] == state[4-i-3][i+3]:
                return 1 if state[i][col]==self.my_piece else -1
        
        # check diamond wins
        for i in range(0,3):
            for j in range(1,4):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j-1] == state[i+2][j] == state[i+1][j+1] and state[i-1][j] == ' ':
                    return 1 if state[i][col]==self.my_piece else -1
                
        return 0 # no winner yet
    def succ(self, state, marker):
        succ = []
        drop_phase = True
        #AI's turn
        if marker == 'AI':
            piece = self.my_piece
            num_chess = 0
            for i in range(5):
                for j in range(5):
                    if not state[i][j] == ' ':
                        num_chess += 1
            if num_chess >= 8:
                drop_phase = False
            if drop_phase:
                for i in range(5):
                        for j in range(5):
                            if state[i][j] == ' ':
                                newState = copy.deepcopy(state)
                                newState[i][j] = piece
                                succ.append(newState)
                return succ

            else:
                for i in range(5):
                    for j in range(5):
                        if state[i][j]==piece:
                            succ+=self.get_succ(state,i,j,piece)
                return succ

        else:
            piece = self.opp
            num_chess = 0
            for i in range(5):
                for j in range(5):
                    if not state[i][j] == ' ':
                        num_chess += 1
            if num_chess >= 8:
                drop_phase = False
            if drop_phase:
                for i in range(5):
                        for j in range(5):
                            if state[i][j] == ' ':
                                newState = copy.deepcopy(state)
                                newState[i][j] = piece
                                succ.append(newState)
                return succ

            else:
                for i in range(5):
                    for j in range(5):
                        if state[i][j]==piece:
                            succ+=self.get_succ(state,i,j,piece)
                return succ
            
    def get_succ(self, state, row, column,piece):
        res = []
        if not row == 0 and state[row-1][column] == ' ':
            newState = copy.deepcopy(state)
            newState[row-1][column] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not row == 4 and state[row+1][column] == ' ':
            newState = copy.deepcopy(state)
            newState[row+1][column] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not column == 0 and state[row][column-1] == ' ':
            newState = copy.deepcopy(state)
            newState[row][column-1] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not column == 4 and state[row][column+1] == ' ':
            newState = copy.deepcopy(state)
            newState[row][column+1] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not row == 0 and not column == 0 and state[row-1][column-1] == ' ':
            newState = copy.deepcopy(state)
            newState[row-1][column-1] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not row == 4 and not column == 0 and state[row+1][column-1] == ' ':
            newState = copy.deepcopy(state)
            newState[row+1][column-1] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not row == 4 and not column == 4 and state[row+1][column+1] == ' ':
            newState = copy.deepcopy(state)
            newState[row+1][column+1] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)
        elif not row == 0 and not column == 4 and state[row-1][column+1] == ' ':
            newState = copy.deepcopy(state)
            newState[row-1][column+1] = self.my_piece
            newState[row][column] = ' '
            res.append(newState)                   
        return res
    
    def heuristic_game_value(self, state,player):
        if not self.game_value(state) == 0 :
            return self.game_value(state)
        # choose player 
        if  player=='AI':
            piece = self.my_piece
            heuristic = self.heuristic_helper(state,piece)
        else:
            piece = self.opp
            heuristic = -self.heuristic_helper(state,piece)
        return heuristic
    
    def heuristic_helper(self, state, piece):
        value=0
        for row in range(5):
            for col in range(5):
                if state[row][col]==piece:
                    if col<=2 and state[row][col+1]==piece and state[row][col+2]==piece:
                        if col<=1 and state[row][col+3]==' ':
                            value+=0.1
                        if col>=1 and state[row][col-1]==' ':
                            value+=0.1   
        return value
    
    def max_value(self,state,depth,marker):
        if self.game_value(state)!=0:
            return self.game_value(state),state
        value=-999
        if depth==0:#base case
            value = self.heuristic_game_value(state, marker)
            return value,state

        succ_list=self.succ(state,marker)
        if marker == 'AI':
            opp = 'opp'
        else:
            opp = 'AI'
        for succes in succ_list:
            min_score=self.min_value(succes,depth-1,opp)[0]
            value=max(value,min_score)
            if value==min_score:
                next_state=succes
        return value,next_state

    def min_value(self,state,depth,marker):
        if self.game_value(state)!=0:
            return self.game_value(state),state
        value = 999
        if depth == 0:#base case
            value = self.heuristic_game_value(state, marker)
            return value, state

        succ_list = self.succ(state, marker)
        if marker == 'AI':
            opp = 'opp'
        else:
            opp = 'AI'
        for succes in succ_list:
            max_score=self.max_value(succes, depth - 1, opp)[0]
            value = min(value, max_score)
            if value == max_score:
                next_state = succes
        return value, next_state
############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()

    
