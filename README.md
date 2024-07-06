# CNN_rock_paper_scissor

A player vs computer rock paper scissor game that identifies the player's hand movements using the MobileNET neural network and plays a random move of its own, as well as keep score. 

## Gather Images

Run `gather_images.py` in terminal with the command `python gather_images.py rock 1000` to open video capture. Press a to start/pause collecting images of your hand and q to quit. You can give any integer argument to collect as many images as you want.
Repeat the process for paper, scissors and none (images of empty background).

## Train data

Run `RPS.py` to train and save a MobileNET model on the created dataset. 

## Play game

Run `Basic_project.py` to play the game. Press escape to quit the game or r to reset the game.
