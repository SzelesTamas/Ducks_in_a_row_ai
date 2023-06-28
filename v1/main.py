from typing import Any, Literal
from ducks_in_a_row import Board
import pygame

class pygameBoard(Board):
    def __init__(self, state=None, startingPlayer: Literal[1, 2] = 1, screen: Any = None, linewidthPercent: float = 0.01):
        super().__init__(state, startingPlayer)
        
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.outsideSquareSize = min(self.width, self.height) / 5
        self.innerSquareSize = self.outsideSquareSize * (1 - linewidthPercent * 2)
        self.linewidthPercent = linewidthPercent
        self.boardRect = [[0 for y in range(5)] for x in range(5)]
        self.selectedSquare = None
        self.font = pygame.font.SysFont('Comic Sans MS', 40)
        
    def drawBoard(self):
        """Draws the board"""
        
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.width, self.height), width=0)
        for x in range(5):
            for y in range(5):
                self.boardRect[x][y] = pygame.draw.rect(self.screen, (176,224,230), (y * self.outsideSquareSize + self.outsideSquareSize * self.linewidthPercent, x * self.outsideSquareSize + self.outsideSquareSize * self.linewidthPercent, self.innerSquareSize, self.innerSquareSize), width=0)
        
    def drawPieces(self):
        """Draws the pieces on the board as circles"""
        
        for x in range(5):
            for y in range(5):
                if(self.state[x][y] == 1):
                    color = (0, 0, 0)
                elif(self.state[x][y] == 2):
                    color = (255, 255, 255)
                else:
                    color = None
                    
                if(color is not None):
                    pygame.draw.circle(self.screen, color, (y * self.outsideSquareSize + self.outsideSquareSize / 2, x * self.outsideSquareSize + self.outsideSquareSize / 2), self.innerSquareSize / 2 * 0.8, width=0)     
               
    def checkGameOver(self):
        """Checks if the game is over and stops the game if it is"""
        
        ret = self.gameOver()
        if(ret == 1):
            text = self.font.render('Player 1(black) wins!', False, (0, 0, 0), (255, 255, 255))
            self.screen.blit(text, (0, 0))
        elif(ret == 2):
            text = self.font.render('Player 2(white) wins!', False, (0, 0, 0), (255, 255, 255))
            self.screen.blit(text, (0, 0))
            
    def draw(self):
        self.drawBoard()
        self.drawPieces()
        self.checkGameOver()
    
    def getSquare(self, pos):
        """Gets the square at a position returns None if no square is found"""
        for x in range(5):
            for y in range(5):
                if(self.boardRect[x][y].collidepoint(pos)):
                    return (x, y)
        return None
       
    def checkClick(self, pos):
        """checks if a square is clicked and sets it as the selected square

        Args:
            pos: mouse position
        """
        
        # get the clicked square
        square = self.getSquare(pos)
        if(square is not None):
            self.selectedSquare = square
        else:
            self.selectedSquare = None
        
    def checkRelease(self, pos):
        """checks where the mouse is released and moves the selected square to that position if it is a valid move

        Args:
            pos: mouse position
        """
        
        if(self.selectedSquare is None):
            return
        square = self.getSquare(pos)
        if(square is not None):
            validMoves = self.getMoves()
            if((self.selectedSquare[0], self.selectedSquare[1], square[0], square[1]) in validMoves):
                self.move(self.selectedSquare[0], self.selectedSquare[1], square[0], square[1], inPlace=True)
        self.selectedSquare = None


pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((500, 500))

board = pygameBoard(screen=screen)
running = True
mousePressed = False
while running:
    # if the user clicks the close button
    for event in pygame.event.get():
        if(event.type == pygame.QUIT):
            running = False
        if(event.type == pygame.MOUSEBUTTONDOWN and not mousePressed):
            mousePressed = True
            board.checkClick(pygame.mouse.get_pos())
        if(event.type == pygame.MOUSEBUTTONUP):
            mousePressed = False
            board.checkRelease(pygame.mouse.get_pos())
            
    board.draw()
    pygame.display.update()
            
    
