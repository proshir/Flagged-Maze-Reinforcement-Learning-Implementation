from enum import Enum
from time import sleep
import pygame as pg
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
0: Way
1: Block
2: Start
3: Target
4: Flag
"""
class Maze():
    def __init__(self, maze, doDraw):
        self.WIDTH = 500
        self.maze = maze
        self.HEIGHT = 550
        self.doDraw = doDraw
        if doDraw:
            self.create_game()
        
        self.color_Way = (255,255,255)
        self.color_GR = (40,10,60)
        color_Block = (255,0,0)
        color_Start = (200,200,200)
        self.color_Black = (0,0,0)
        color_Flag = (0,160,90)
        color_Target = (200,200,0)

        self.dict_color = {0:self.color_Way, 1:color_Block, 2:color_Start, 3:color_Target, 4: color_Flag}
        self.border_width = 1
        self.cell_size = self.WIDTH/len(maze)

        self.flags = {}
        self.orderedFlags = []

        for y in range(len(maze)):
            for x in range(len(maze[y])):
                if maze[y][x] == 2:
                    self.start_pos = [y, x]
                elif maze[y][x] == 4:
                    self.flags[(y, x)] = False
                    self.orderedFlags += [(y,x)]

        self.player = Player(self)
        self.steps = 0

    def create_game(self):
        pg.init()
        self.font = pg.font.Font('FreeSansBold.ttf', 32)
        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        pg.display.set_caption("MAZE RL")
        
    def reset(self):
        self.player.reset(self)
        self.steps = 0
        self.watched = {self.player.player_pos}
        for x in self.flags.keys():
            self.flags[x] = False

    def capture_flag(self, y, x):
        if self.maze[y][x] == 4 and self.flags[(y,x)] == False:
            self.flags[(y,x)] = True
            return True
        return False

    def draw(self):
        self.screen.fill(self.color_Black)
        for y in range(len(self.maze)):
            for x in range(len(self.maze[y])):
                pg.draw.rect(self.screen, self.color_GR, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), self.border_width)
                
                if self.maze[y][x] != 4 or self.flags[(y,x)] == False:
                        color = self.dict_color[self.maze[y][x]]
                else:
                    color = self.dict_color[0]

                pg.draw.rect(self.screen, color, (x * self.cell_size + self.border_width, y * self.cell_size + self.border_width, self.cell_size - 2 * self.border_width, self.cell_size - 2 * self.border_width))
        self.player.draw(self)
        self.text_render()
        pg.display.update()

    def text_render(self):
        text = self.font.render("Steps: {}".format(self.steps), True, self.color_Way)
        textRect = text.get_rect()
        textRect.center = 100 , 520
        self.screen.blit(text, textRect)

    def check_visited(self, y, x):
        flag = False
        if (y,x) in self.watched:
            flag = True
        else:
            self.watched.add((y,x))
        return flag

    def move(self, action):
        y,x = self.player.move(action, True)
        self.steps += 1
        if self.check_visited(y, x):
            return -10, False
        if self.capture_flag(y,x):
            return 50, False
        if self.maze[y][x] == 3:
            if False in self.flags.values():
                return -400, True
            return 100, True
        return -1, False

    def get_flags(self, only_count):
        if only_count:
            flags_count = 0
            for value in self.flags.values():
                if value:
                    flags_count += 1
            return flags_count
        return tuple([self.flags[(y, x)] for y, x in self.orderedFlags])

    def check_cell_valid(self, y, x):
        return not(y < 0 or x < 0 or y >= len(self.maze) or x >= len(self.maze[0]) or self.maze[y][x] == 1)

class Player():
    def __init__(self, maze):
        self.reset(maze)
        self.color = (100,50,100)
        self.radius = maze.cell_size * .4

    def reset(self, maze):
        self.player_pos = maze.start_pos[0], maze.start_pos[1]

    def draw(self, maze):
        y, x = (self.player_pos[0] + .5) * maze.cell_size, (self.player_pos[1] + .5) * maze.cell_size
        pg.draw.circle(maze.screen, self.color, (x, y), self.radius)

    def check_move_valid(self, maze, action):
        y, x = self.move(action)
        return maze.check_cell_valid(y, x)

    def move(self, action, assign=False):
        y, x = self.player_pos[0], self.player_pos[1]
        if action == Action.Up:
            y -= 1
        elif action == Action.Down:
            y += 1
        elif action == Action.Right:
            x += 1
        else:
            x -= 1
        if assign:
            self.player_pos = y, x
        return y, x

class Action(Enum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3
    Default = 6

class RL():
    def __init__(self, maze, times, gamma= .9, alpha= .1, epsilon= .1, only_count=False):
        self.maze = maze
        self.player = maze.player
        self.actions = [Action.Up, Action.Right, Action.Down, Action.Left]
        self.times = times
        self.gamma = gamma
        self.Qs = {}
        self.alpha = alpha
        self.epsilon = epsilon 
        self.only_count = only_count

    def choose(self, state, playing= False):
        possible_actions = self.get_possible_actions()
        if not playing and random.random() < self.epsilon:
            return random.choice(possible_actions)
        Qs = [self.getQ(state, a) for a in possible_actions]
        maxQ = max(Qs)
        bigest = random.choice([a for i, a in enumerate(possible_actions) if Qs[i] == maxQ])  
        return bigest

    def get_possible_actions(self):
        return [action for action in self.actions if self.player.check_move_valid(maze, action)]

    def getQ(self, state, action):
        if (state, action) in self.Qs:
            return self.Qs[(state, action)]
        return 0

    def updateQ(self, state, action, reward, next_state):   
        Q = self.getQ(state, action)
        maxQnext = 0
        possible_moves = self.get_possible_actions()
        for a in possible_moves:
            maxQnext = max(self.getQ(next_state, a), maxQnext)
        if len(possible_moves):
            Q = Q + self.alpha * (reward + self.gamma * maxQnext - Q)
        else :
            Q += reward
        self.Qs[(state, action)] = Q 
        return Q

    def get_state(self):
        return self.player.player_pos[0], self.player.player_pos[1], self.maze.get_flags(self.only_count)

    def start(self):
        self.steps = np.zeros(self.times)
        self.costs = np.zeros(self.times)
        for episode in range(self.times):
            step, cost = self.play(self.maze.doDraw)
            print("Done {0}| Steps: {1} with Costs: {2}".format(episode, step, cost))
            self.costs[episode] = cost
            self.steps[episode] = self.maze.steps
        print("The average of the last 5: ",np.average(self.steps[-5:]))
    
    def print_Qs_Nan(self):
        data = [(k[0], k[1].name, v) for k, v in rl.Qs.items()]
        df = pd.DataFrame.from_records(data, columns=['State', 'Action', 'Value'])
        df.set_index(['State', 'Action'], inplace=True)
        df = df.unstack()
        df.columns = df.columns.get_level_values(1)
        df.reset_index(inplace=True)
        df.index.name = None
        pd.set_option('display.max_rows', df.shape[0])
        return df
    
    def print_Qs(self):
        df = self.print_Qs_Nan()
        df.fillna(0, inplace=True)
        return df

    def plt_steps(self):
        plt.figure()
        plt.plot(np.arange(len(self.steps)), self.steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
    
    def plt_costs(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs, 'b')
        plt.title('Episode via costs')
        plt.xlabel('Episode')
        plt.ylabel('Costs')

    def plt_costs_av_steps(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs/self.steps, 'b')
        plt.title('Episode via average costs over steps')
        plt.xlabel('Episode')
        plt.ylabel('average costs over steps')

    def play(self, doDraw=True, playing=False):
        cost = 0
        self.maze.reset()
        state = self.get_state()
        if playing and doDraw:
            self.maze.create_game()
            sleep(5)
        while self.maze.steps < 3000:
            if doDraw and playing:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        return
                maze.draw()
                if playing:
                    sleep(.2)
            action = self.choose(state, playing)
            reward, target = maze.move(action)
            next_state = self.get_state()
            cost += self.updateQ(state, action, reward, next_state)
            state = next_state
            if target:
                if doDraw:
                    maze.draw()
                    sleep(1)
                    if playing:
                        pg.quit()
                return self.maze.steps, cost 
        if playing and doDraw:
            pg.quit()
        return self.play(doDraw, playing)

if __name__ == '__main__':       
    mazeM = [
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 4, 1, 4, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 3, 1, 0, 0, 1, 1, 1, 1]]
    maze = Maze(mazeM, True)
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                break
        maze.draw()
    # doDraw = False
    # maze = Maze(mazeM, doDraw)
    # rl = RL(maze, 1000)
    # rl.start()
    # if doDraw:
    #     pg.quit()