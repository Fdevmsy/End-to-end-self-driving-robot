import numpy as np
import pandas as pd
import pygame
import glob
import os
# from config import VisualizeConfig

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

# config = VisualizeConfig()
preds = pd.read_csv('scripts/dataset/predict.csv', names=['preds_angles'])
dataset = pd.read_csv('scripts/dataset/dataset.csv', names=['center', 'speed', 'angle'])
# filenames = glob.glob(config.img_path)
# print(dataset["center"].values)
pygame.init()
size = (320, 240)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

# for i in range(1000):
for i in range(len(dataset)):

    screen.fill((RED))
    angle = preds["preds_angles"].iloc[i] # radians
    true_angle = dataset["angle"].iloc[i] # radians
    
    # add image to screen
    img = pygame.image.load(os.path.join('scripts/dataset/', dataset["center"].iloc[i]))
    screen.blit(img, (0, 0))
    
    # add text
    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    screen.blit(pred_txt, (10, 200))
    screen.blit(true_txt, (10, 220))

    # draw steering wheel
    radius = 40
    pygame.draw.circle(screen, WHITE, [240, 200], radius, 2) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [240 + int(x), 200 - int(y)], 7)
    
    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, BLACK, [240 + int(x), 200 - int(y)], 5) 
    

    pygame.display.update()
    pygame.display.flip()