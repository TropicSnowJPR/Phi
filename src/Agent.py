# Standard library
import json
import os
import socket
import sys
import threading
import warnings
from collections import deque
from queue import Queue
import time

# Third-party
import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn

# Local modules
from CNN import CNN
from GameController import GameController
from GameDataReader import GameDataReader
from LSTM import Memory
from Policy import Policy
from RLDatabase import RLDatabase


class Phi:
    
    def __init__(self):

        if torch.cuda.is_available():
            self.DEVICE = "cuda"
            torch.backends.nnpack.enabled = True
        else:
            self.DEVICE = "cpu"
            torch.backends.nnpack.enabled = False

        with open("config.json", "r") as file:
            self.CONFIG = json.load(file)
            print(self.CONFIG)
        
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        try:
            self.CHECKPOINT_DIR = os.path.join(self.PROJECT_ROOT, self.CONFIG["checkpoint_dir"])
            os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
            self.CHECKPOINT_PATH = os.path.join(
                self.CHECKPOINT_DIR,
                self.CONFIG["checkpoint_file"] + ".pth"
            )

            self.DATABASE_DIR = os.path.join(self.PROJECT_ROOT, self.CONFIG["database_dir"])
            os.makedirs(self.DATABASE_DIR, exist_ok=True)
            self.DATABASE_PATH = os.path.join(
                self.DATABASE_DIR,
                self.CONFIG["database_file"] + ".db"
            )

        except OSError as e:
            raise Exception(f"Invalid path configuration: {e}") from e
        
            
        self.ACTIONS = []
                      
        for action in self.CONFIG["actions"]:
            self.ACTIONS.append(action["action"])
        
        self.DBMANAGER = RLDatabase(self.DATABASE_PATH)

        class Agent(nn.Module):
            def __init__(self, MEMORY_INPUT_SIZE, MEMORY_HIDDEN_SIZE, POLICY_INPUT_SIZE, POLICY_ACTION_SIZE):
                super().__init__()
                
                self.CNN = CNN()
                self.LSTM = Memory( input_size = MEMORY_INPUT_SIZE, hidden_size = MEMORY_HIDDEN_SIZE )
                self.POLICY = Policy( input_size = POLICY_INPUT_SIZE, action_size = POLICY_ACTION_SIZE )
                
                
            def forward(self, INPUT, HIDDEN):
                BATCH, TIME, CHANNELS, HEIGHT, WIDTH = INPUT.shape

                INPUT = INPUT.view(BATCH * TIME, CHANNELS, HEIGHT, WIDTH)
                FEATURES = self.CNN(INPUT)

                FEATURES = FEATURES.view(BATCH, TIME, -1)

                LSTM_OUT, HIDDEN = self.LSTM(FEATURES, HIDDEN)

                LGOITS = self.POLICY(LSTM_OUT[:, -1])

                return LGOITS, HIDDEN
            
            
        model = CNN()
        x = torch.zeros(1, 3, 84, 84)
        CNN_OUTPUT_SIZE = model(x).shape[1]
            
        self.AGENT = Agent(
            MEMORY_INPUT_SIZE = CNN_OUTPUT_SIZE, 
            MEMORY_HIDDEN_SIZE = 256, 
            POLICY_INPUT_SIZE = 256, 
            POLICY_ACTION_SIZE = len(self.ACTIONS) 
        )
        
        self.AGENT.to( device = self.DEVICE)
        self.OPTIMIZER = optim.Adam( params = self.AGENT.parameters(), lr=1e-4 )
        
        self.HIDDEN = ( torch.zeros( 1, 1, 256, device = self.DEVICE ), torch.zeros(1, 1, 256, device = self.DEVICE ))
        
        self.MEMORY = deque( maxlen = 32 )
        
        self.STATE = deque( maxlen=1 )
        self.STATE_QUEUE = deque( maxlen=2 )


        self.RUNNING = True

        self.TCP_SERVER = None
        
        self.TCP_SERVER_THREAD = threading.Thread( target = self.tcp_state_server, daemon=True )
        self.TCP_SERVER_THREAD.start()
        
        
        self.REWARD_BASELINE = 0.0

        
        self.STEPS = 0
        
        self.EPISODE_ID = None
        self.EPISODE_START_TIME = None
        self.EPISODE_TOTAL_REWARD = 0.0

        
        self.LAST_TICK = -1
    
    
        self.GAME_CONTROLLER = GameController()
        
        
        
    def tcp_state_server( self, HOST = "0.0.0.0", PORT = 8001 ):
        self.TCP_SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.TCP_SERVER.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCP_SERVER.bind((HOST, PORT))
        self.TCP_SERVER.listen(1)

        print("Waiting for Minecraft mod connection...")

        CONN, ADDR = self.TCP_SERVER.accept()
        print("Connected:", ADDR)

        BUFFER = ""

        while True:
            DATA = CONN.recv(65536)
            if not DATA:
                print("Client disconnected")
                break

            BUFFER += DATA.decode("utf-8")

            while "\n" in BUFFER:
                try:

                    LINE, BUFFER = BUFFER.split("\n", 1)
                    STATE = json.loads(LINE)
                    self.STATE.append(STATE)

                except json.JSONDecodeError:
                    print("TCP ERROR:", repr(e))
                    continue
                except Exception as e:
                    print(e)
                


    def select_action(self, logits):
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        return action.item()

    def preprocess(self, frame):
        frame = cv2.resize(frame, (84, 84))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0

        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)

        return frame

    def get_minecraft_frame(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = np.array(sct.grab(monitor))
            return img[:, :, :3]  # remove alpha
        return None

    def take_action_in_game(self, action):

        raw = self.get_latest_state()
        while raw is None:
            time.sleep(5)
            raw = self.get_latest_state()
            

        current_state = GameDataReader(raw)

        self.STATE_QUEUE.append(current_state)

        reward = 0

        prev_state = self.STATE_QUEUE[0] if len(self.STATE_QUEUE) > 1 else current_state
        reward = self.get_reward(prev_state, current_state)

        if action == 0:
            pass

        elif action == 1:
            self.GAME_CONTROLLER.press_key("w", 100)

        elif action == 2:
            self.GAME_CONTROLLER.press_key("a", 100)

        elif action == 3:
            self.GAME_CONTROLLER.press_key("d", 100)

        elif action == 4:
            self.GAME_CONTROLLER.press_key("s", 100)

        elif action == 5:
            self.GAME_CONTROLLER.press_key("space", 100)

        elif action == 6:
            self.GAME_CONTROLLER.move_smooth(10, 0, 10, 1 / 60)

        elif action == 7:
            self.GAME_CONTROLLER.move_smooth(-10, 0, 10, 1 / 60)

        return reward, current_state

    def get_reward(self, prev, curr):
        reward = 0.0

        if curr.position[0] != prev.position[0] or curr.position[2] != prev.position[2]:
            reward += (abs(curr.position[2] - prev.position[2]) + abs(curr.position[0] - prev.position[0])) * 0.05

        reward -= 0.5 * (prev.health - curr.health)

        # idle penalty ( jump is ignored since it can because the AI to jump and not move )
        if curr.position[0] == prev.position[0] and curr.position[2] == prev.position[2]:
            reward -= 0.2

        if curr.position[1] > prev.position[1]:
            pass

        reward += 0.05

        return reward

    def get_next_episode_index(self):
        row = self.DBMANAGER.fetchone(
            "SELECT MAX(id) AS max_id FROM Episodes"
        )

        if row is None or row["max_id"] is None:
            return 0

        return row["max_id"] + 1

    def episode_done(self, state):
        if state.health == 0:
            # Respawn Button Click
            self.GAME_CONTROLLER.press_key("tab", 10)
            self.GAME_CONTROLLER.press_key("enter", 10)
        return state.health == 0
    
    def start_episode(self):
        import time

        self.EPISODE_START_TIME = int(time.time())
        self.EPISODE_TOTAL_REWARD = 0.0

        next_index = self.get_next_episode_index()

        self.EPISODE_ID = self.DBMANAGER.execute(
            """
            INSERT INTO Episodes (
                episode_index,
                start_time,
                total_steps,
                total_reward,
                termination_reason
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                next_index,
                self.EPISODE_START_TIME,
                0,
                0.0,
                None
            )
        )

        if self.EPISODE_ID is None:
            raise RuntimeError("Failed to create episode")
        
    def log_step(self, action, reward, state, log_prob=None, advantage=None, baseline=None):
        step_id = self.DBMANAGER.execute(
            """
            INSERT INTO Steps (
                episode_id,
                tick,
                timestamp,
                action,
                reward,

                x_position,
                y_position,
                z_position,
                yaw,
                pitch,

                health,
                food,
                armor,
                xp_level,

                inventory,
                target,

                game_time,
                raining,
                thundering,
                difficulty,

                entities,
                blocks
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.EPISODE_ID,
                state.tick,
                int(time.time()),
                action,
                reward,

                state.position[0],
                state.position[1],
                state.position[2],
                state.rotation[0],
                state.rotation[1],

                state.health,
                state.food,
                state.armor,
                state.xp_level,

                json.dumps(state.inventory),
                json.dumps(state.target_block),

                state.game_time,
                int(state.is_raining),
                int(state.is_thundering),
                state.difficulty,

                json.dumps(state.entities),
                json.dumps(state.blocks)
            )
        )

        if log_prob is not None:
            self.DBMANAGER.execute(
                """
                INSERT INTO StepDiagnostics (
                    step_id,
                    log_prob,
                    advantage,
                    baseline
                ) VALUES (?, ?, ?, ?)
                """,
                (step_id, log_prob, advantage, baseline)
            )

        return step_id

    def end_episode(self, reason):
        import time

        self.DBMANAGER.execute(
            """
            UPDATE Episodes
            SET end_time = ?,
                total_steps = ?,
                total_reward = ?,
                termination_reason = ?
            WHERE id = ?
            """,
            (
                int(time.time()),
                self.STEPS,
                self.EPISODE_TOTAL_REWARD,
                reason,
                self.EPISODE_ID
            )
        )

    def get_latest_state(self):
        if len(self.STATE) == 0:
            return None

        state = self.STATE[-1]

        if state is None:
            return None

        if state.get("tick", -1) == self.LAST_TICK:
            return None

        self.LAST_TICK = state.get("tick", -1)
        return state

    def save_checkpoint(self, agent, optimizer, step, hidden_state):
        torch.save({
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "hidden": (
                hidden_state[0].cpu(),
                hidden_state[1].cpu()
            ),
        }, self.CHECKPOINT_PATH)

    def startup(self):
        print("Starting Minecraft agent...")

        self.GAME_CONTROLLER.press_key("t", 10)
        self.GAME_CONTROLLER.press_key("esc", 10)
        self.GAME_CONTROLLER.press_key("t", 10)
        self.GAME_CONTROLLER.press_key("up", 10)
        self.GAME_CONTROLLER.press_key("enter", 10)

    def loop(self): 
        try:            
            if self.EPISODE_ID is None:
                self.start_episode()
                
            self.STEPS += 1

            # OBSERVE
            FRAME = self.get_minecraft_frame()
            FRAME = self.preprocess(FRAME)

            FRAME = FRAME.unsqueeze(0).unsqueeze(0)  # (1,1,3,84,84)
            FRAME = FRAME.to( self.DEVICE )

            # DETACH MEMORY
            self.HIDDEN = ( self.HIDDEN[0].detach(), self.HIDDEN[1].detach() )

            # FORWARD PASS
            LOGITS, self.HIDDEN = self.AGENT( FRAME, self.HIDDEN )

            # ACTION
            self.ACTION = self.select_action( LOGITS )

            # ENV STEP
            REWARD, CURRENT_STATE = self.take_action_in_game( self.ACTION )
            
            self.REWARD_BASELINE = 0.99 * self.REWARD_BASELINE + 0.01 * REWARD
            ADVANTAGE = REWARD - self.REWARD_BASELINE
            
            self.EPISODE_TOTAL_REWARD += REWARD
            
            # LEARNING (REINFORCE)
            LOG_PROBS = torch.log_softmax( LOGITS, dim=-1 )

            LOG_PROB = LOG_PROBS[0, self.ACTION]
            
            step_id = self.log_step(
                action=self.ACTION,
                reward=REWARD,
                state=CURRENT_STATE,
                log_prob=LOG_PROB.detach().cpu().item(),
                advantage=ADVANTAGE.detach().cpu().item() if torch.is_tensor(ADVANTAGE) else ADVANTAGE,
                baseline=self.REWARD_BASELINE
            )

            self.MEMORY.append((LOG_PROB, ADVANTAGE))
            
            if len(self.MEMORY) == 32:
                loss = 0

                for log_prob, advantage in self.MEMORY:
                    loss += -log_prob * advantage

                self.OPTIMIZER.zero_grad()
                loss.backward()
                self.OPTIMIZER.step()

                self.MEMORY.clear()

            # 7. RESET EPISODE
            if self.episode_done(CURRENT_STATE) or self.STEPS >= 2000:
                self.end_episode(
                    reason="death" if CURRENT_STATE.health == 0 else "timeout"
                )
                self.HIDDEN = ( 
                    torch.zeros( 1, 1, 256, device = self.DEVICE ), 
                    torch.zeros(1, 1, 256, device = self.DEVICE )
                )
                self.STEPS = 0
                self.REWARD_BASELINE = 0.0
                self.MEMORY.clear()
                self.save_checkpoint(self.AGENT, self.OPTIMIZER, self.STEPS, self.HIDDEN)
                self.EPISODE_ID = None
            
            print("Action:", self.ACTIONS[self.ACTION], "Reward:", REWARD)
        except Exception as e:
            raise e



if __name__ == "__main__":
    try:
        PhiAgent = Phi()
        PhiAgent.startup()
        while PhiAgent.RUNNING:
            PhiAgent.loop()
    except KeyboardInterrupt:
        
        PhiAgent.end_episode(
            "timeout"
        )
        
        HIDDEN = ( 
            torch.zeros( 1, 1, 256, device = PhiAgent.DEVICE ), 
            torch.zeros( 1, 1, 256, device = PhiAgent.DEVICE )
        )

        PhiAgent.save_checkpoint(PhiAgent.AGENT, PhiAgent.OPTIMIZER, PhiAgent.STEPS, HIDDEN)
        
        PhiAgent.GAME_CONTROLLER.press_key("esc", 10)
        
        PhiAgent.RUNNING = False
        
        if PhiAgent.TCP_SERVER is not None:
            try:
                PhiAgent.TCP_SERVER.close()
            except Exception:
                pass
            
        try:
            PhiAgent.TCP_SERVER_THREAD.join( timeout=2 )
        except Exception:
            pass

        print("Shutdown complete. Exiting.")
        sys.exit(0)
    
    except Exception as e:
        print(e)
    
        
# TODO: Improve exploration and movement incentives
#
# - Add a penalty when the agent remains in the same area for an extended period.
# - Increase jump-related rewards to better encourage vertical movement and
#   climbing onto higher blocks.
# - Consider rewarding useful action sequences, such as:
#     Jump -> Move -> Successful elevation gain
#   This could provide a larger reward than a standalone jump.
#
# - Detect nearby points of interest (e.g. special blocks, structures, ores)
#   and grant a one-time exploration bonus when first discovered.
#
# - Implement an anti-camping system:
#     - Spawn invisible "stagnation markers" in areas where the agent spends
#       excessive time.
#     - Nearby markers reduce rewards earned in that region.
#     - Marker influence should gradually decay over time.
#     - Markers should be removed after sufficient decay or at the end of an
#       episode.
#
# - Investigate imitation-learning pretraining:
#     - Record a short demonstration episode from a human player.
#     - Use the collected observations/actions to initialize agent behavior.
#     - Allow the agent to learn its own internal representation of useful
#       movement, exploration, and interaction patterns before reinforcement
#       learning begins.