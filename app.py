
import os, glob, random
import numpy as np 
import streamlit as st 
from PIL import Image
from tensorflow import keras 

N_ROUNDS =  10

@st.cache_resource 
def load_model():
  return keras.models.load_model("best_model.keras")

model = load_model()

# Test images: 1 = Real, 0 = AI 

images = []
for path in sorted(glob.glob("test_images/*.jpg")):
  if os.path.basename(path).startswith("real"):
    label = 1
  else:
    label = 0
  images.append((path, label))



def model_predict(path):
  img = Image.open(path).convert("RGB").resize((128, 128))
  # MobileNet expects pixels in [0, 255]
  arr = np.array(img, dtype="float32")
  # Making it a batch of 1 image: (1, 128, 128, 3)      
  batch = np.expand_dims(arr, axis=0)
  # model output, shape (1, 1), e.g. [[0.87]]       
  prediction = model.predict(batch, verbose=0)  
  # taking the single number out: 0.87 
  score = prediction[0][0]                  
  return float(score)                        

# Game Functions

def new_game():
  # pick N_ROUNDS random images for this game
  st.session_state.order = random.sample(range(len(images)), N_ROUNDS)  
  # current round (starts at 0)
  st.session_state.round = 0
  # human score          
  st.session_state.human = 0   
  # model score       
  st.session_state.model = 0 
  # has the user guessed this round?         
  st.session_state.answered = False   
  # feedback for this round
  st.session_state.result = None     

def guess(choice):
  #The current image and its correct answer
  index = st.session_state.order[st.session_state.round]
  path, truth = images[index]

  # The model makes its prediction
  score = model_predict(path)
  if score >= 0.5:
    model_choice = 1
  else:
    model_choice = 0

  # Update the scores
  if choice == truth:
    st.session_state.human += 1
  if model_choice == truth:
    st.session_state.model += 1

  st.session_state.answered = True
  st.session_state.result = (truth, choice, model_choice, score)

def next_image():
  st.session_state.round += 1
  st.session_state.answered = False
  st.session_state.result = None 
#Start a game the first time the app opens 
if "order" not in st.session_state:
  new_game()
names = {1: "Real", 0: "AI-generated"}
is_last_round = st.session_state.round == N_ROUNDS - 1
game_over = st.session_state.answered and is_last_round

# Number of rounds played so far
played = st.session_state.round
if st.session_state.answered:
  played += 1

# Title and Instuctions
st.title("Real or AI? Human vs. Model")
st.write(f"Look at each face and guess if it is Real or AI-generated. "
         f"The model plays too. After {N_ROUNDS} rounds, the higher score wins!")

# Rounds counter and  live scores 
st.subheader(f"Round {st.session_state.round + 1} / {N_ROUNDS}")
st.write(f"**You:** {st.session_state.human} / {played}   |   "
         f"**Model:** {st.session_state.model} / {played}")

# Image display
index = st.session_state.order[st.session_state.round]
st.image(images[index][0], width=300)

# Real or AI- Generated buttons
col1, col2 = st.columns(2)
col1.button("Real", on_click=guess, args=(1,), disabled=st.session_state.answered)
col2.button("AI-Generated", on_click=guess, args=(0,), disabled=st.session_state.answered)

# Feedback: Revealing the correct answer

if st.session_state.result is not None:
  truth, choice, model_choice, score = st.session_state.result
  message = (f"Correct answer: **{names[truth]}**  \n"
             f"You said: {names[choice]}  \n"
             f"Model said: {names[model_choice]} (score {score:.2f})")
  if choice == truth:
    st.success("Correct! " + message)     
  else:
    st.error("Wrong! " + message)      
  
# Next Image Button
if st.session_state.answered and not is_last_round:
  st.button("Next Image", on_click=next_image)


# Final summary
if game_over:
  human = st.session_state.human
  model_score = st.session_state.model

  if human > model_score:
    st.balloons()
    st.success(f"You win! You: {human}/{N_ROUNDS}, Model: {model_score}/{N_ROUNDS}")
  elif human < model_score:
    st.warning(f"The model wins! You: {human}/{N_ROUNDS}, Model: {model_score}/{N_ROUNDS}")
  else:
    st.info(f"It's a tie! You: {human}/{N_ROUNDS}, Model: {model_score}/{N_ROUNDS}")

  st.button("New Game", on_click=new_game)


