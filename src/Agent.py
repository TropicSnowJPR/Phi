import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import mss
import numpy as np

from CNN import CNN
from LSTM import Memory
from Policy import Policy
from GameController import GameController


# ========================
# GLOBAL: CNN output size
# ========================
def get_cnn_output_size():
    model = CNN()
    x = torch.zeros(1, 3, 84, 84)
    return model(x).shape[1]

controller = GameController()


CNN_OUTPUT_SIZE = get_cnn_output_size()


# ========================
# AGENT
# ========================
class MinecraftAgent(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.cnn = CNN()
        self.memory = Memory(input_size=CNN_OUTPUT_SIZE, hidden_size=256)
        self.policy = Policy(256, action_size)

    def forward(self, x, hidden):
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        features = self.cnn(x)

        features = features.view(b, t, -1)

        lstm_out, hidden = self.memory(features, hidden)

        logits = self.policy(lstm_out[:, -1])

        return logits, hidden


# ========================
# MEMORY
# ========================
def init_hidden(batch_size=1):
    return (
        torch.zeros(1, batch_size, 256),
        torch.zeros(1, batch_size, 256)
    )


# ========================
# ACTION SELECTION
# ========================
def select_action(logits):
    probs = torch.softmax(logits, dim=-1)
    action = torch.multinomial(probs, 1)
    return action.item()


# ========================
# SCREEN PROCESSING
# ========================
def preprocess(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32") / 255.0

    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)

    return frame


def get_minecraft_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        return img[:, :, :3]  # remove alpha


# ========================
# GAME INTERACTION (FAKE)
# ========================
def take_action_in_game(action):
    reward = 0.1

    if action == 0:
        pass

    elif action == 1:
        controller.press_key("w", 100)

    elif action == 2:
        controller.press_key("a", 100)

    elif action == 3:
        controller.press_key("d", 100)

    elif action == 4:
        controller.press_key("space", 100)

    elif action == 5:
        controller.click()
        
    return reward


def episode_done():
    return False


# ========================
# MAIN LOOP
# ========================
if __name__ == "__main__":

    action_size = 6

    agent = MinecraftAgent(action_size=action_size)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)

    hidden = init_hidden(batch_size=1)

    ACTIONS = [
        "nothing",
        "forward",
        "left",
        "right",
        "jump",
        "attack"
    ]

    while True:

        # -------------------------
        # 1. OBSERVE
        # -------------------------
        frame = get_minecraft_frame()
        frame = preprocess(frame)

        frame = frame.unsqueeze(0).unsqueeze(0)  # (1,1,3,84,84)

        # -------------------------
        # 2. DETACH MEMORY
        # -------------------------
        hidden = (hidden[0].detach(), hidden[1].detach())

        # -------------------------
        # 3. FORWARD PASS
        # -------------------------
        logits, hidden = agent(frame, hidden)

        # -------------------------
        # 4. ACTION
        # -------------------------
        action = select_action(logits)

        # -------------------------
        # 5. ENV STEP
        # -------------------------
        reward = take_action_in_game(action)

        # -------------------------
        # 6. LEARNING (REINFORCE)
        # -------------------------
        log_probs = torch.log_softmax(logits, dim=-1)

        loss = -log_probs[0, action] * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------
        # 7. RESET EPISODE
        # -------------------------
        if episode_done():
            hidden = init_hidden(batch_size=1)

        print("Action:", ACTIONS[action], "Reward:", reward)