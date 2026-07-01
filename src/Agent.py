# Standard library
import json
import yaml
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

        with open("config.yaml", "r") as file:
            self.CONFIG = yaml.safe_load(file)
        
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        try:
            self.CHECKPOINT_PATH = os.path.join(self.CONFIG["storage"]["checkpoint"]["path"])
            if not self.CHECKPOINT_PATH.endswith(".pth"):
                self.CHECKPOINT_PATH += ".pth"
            os.makedirs(os.path.dirname(self.CHECKPOINT_PATH), exist_ok=True)
            
            self.DATABASE_PATH = os.path.join(self.CONFIG["storage"]["database"]["path"])
            if not self.DATABASE_PATH.endswith(".db"):
                self.DATABASE_PATH += ".db"
            os.makedirs(os.path.dirname(self.DATABASE_PATH), exist_ok=True)
        except OSError as e:
            raise Exception(f"Invalid path configuration: {e}") from e
          
        self.ACTIONS = []
                      
        for action in self.CONFIG["agent"]["action_space"]["actions"]:
            self.ACTIONS.append(action["id"])

        self.SCREEN_WIDTH = self.CONFIG["agent"]["environment"]["screen"]["width"]
        self.SCREEN_HEIGHT = self.CONFIG["agent"]["environment"]["screen"]["height"]
        self.CAPTURE_MONITOR_INDEX = self.CONFIG["agent"]["environment"]["capture"]["monitor_index"]
        self.INPUT_CHANNELS = self.CONFIG["model"]["encoder"]["input_channels"]
        self.LSTM_HIDDEN = self.CONFIG["model"]["temporal"]["hidden_size"]
        self.KEY_HOLD_MS = self.CONFIG["agent"]["action_space"]["defaults"]["key_hold_ms"]
        self.MOUSE_MOVE_MS = self.CONFIG["agent"]["action_space"]["defaults"]["mouse_move_ms"]
        self.STEP_DELAY_S = self.CONFIG["agent"]["action_space"]["defaults"]["step_delay_s"]
        self.TURN_LEFT_DELTA_X = self.CONFIG["agent"]["action_space"]["actions"][6]["delta"]["x"]
        self.TURN_RIGHT_DELTA_X = self.CONFIG["agent"]["action_space"]["actions"][7]["delta"]["x"]
        self.SEQUENCE_LENGTH = self.CONFIG["agent"]["observation"]["memory"]["sequence_length"]
        self.STATE_CURRENT = self.CONFIG["agent"]["observation"]["state_buffer"]["current"]
        self.STATE_PREVIOUS = self.CONFIG["agent"]["observation"]["state_buffer"]["previous"]
        self.OPTIMIZER_LR = self.CONFIG["training"]["optimizer"]["learning_rate"]
        self.BATCH_SIZE = self.CONFIG["training"]["replay_buffer"]["batch_size"]
        self.DISCOUNT_FACTOR = self.CONFIG["reward"]["shaping"]["discount_factor"]
        self.REWARD_PASSIVE = self.CONFIG["reward"]["components"]["passive"]
        self.REWARD_MOVEMENT_PER_BLOCK = self.CONFIG["reward"]["components"]["movement_per_block"]
        self.REWARD_IDLE_PENALTY = self.CONFIG["reward"]["components"]["idle_penalty"]
        self.REWARD_DAMAGE_PER_HEART = self.CONFIG["reward"]["components"]["damage_per_heart"]
        self.MAX_STEPS = self.CONFIG["agent"]["episode"]["max_steps"]
        self.TCP_BUFFER_SIZE = self.CONFIG["network"]["tcp"]["buffer_size"]
        self.TCP_TIMEOUT_S = self.CONFIG["network"]["tcp"]["timeout_ms"] / 1000
        self.SHUTDOWN_TIMEOUT_S = self.CONFIG["runtime"]["shutdown"]["timeout_s"]
            
        if self.CONFIG["agent"]["model"]["device"] == "auto":
            if torch.cuda.is_available():
                self.DEVICE = "cuda"
                torch.backends.nnpack.enabled = True
            else:
                self.DEVICE = "cpu"
                torch.backends.nnpack.enabled = False
        else:
            self.DEVICE = self.CONFIG["agent"]["model"]["device"]
        
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
        x = torch.zeros(1, self.INPUT_CHANNELS, self.SCREEN_HEIGHT, self.SCREEN_WIDTH)
        CNN_OUTPUT_SIZE = model(x).shape[1]
            
        self.AGENT = Agent(
            MEMORY_INPUT_SIZE = CNN_OUTPUT_SIZE, 
            MEMORY_HIDDEN_SIZE = self.LSTM_HIDDEN, 
            POLICY_INPUT_SIZE = self.LSTM_HIDDEN,
            POLICY_ACTION_SIZE = len(self.ACTIONS) 
        )
        
        self.AGENT.to( device = self.DEVICE )
        self.OPTIMIZER = optim.Adam( params = self.AGENT.parameters(), lr=self.OPTIMIZER_LR )
        
        self.HIDDEN = ( torch.zeros( 1, 1, self.LSTM_HIDDEN, device = self.DEVICE ), torch.zeros(1, 1, self.LSTM_HIDDEN, device = self.DEVICE ))
        
        self.MEMORY = deque( maxlen = self.SEQUENCE_LENGTH )
        
        self.STATE = deque( maxlen= self.STATE_CURRENT )
        self.STATE_QUEUE = deque( maxlen= self.STATE_PREVIOUS )


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
        
    def validate_config(cfg):
        assert cfg["agent"]["environment"]["screen"]["width"] > 0
        assert cfg["agent"]["environment"]["screen"]["height"] > 0
        assert cfg["agent"]["model"]["temporal"]["hidden_size"] > 0
        
    def tcp_state_server( self ):
        self.TCP_SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.TCP_SERVER.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.TCP_SERVER.bind((self.CONFIG["network"]["tcp"]["host"], self.CONFIG["network"]["tcp"]["port"]))
        self.TCP_SERVER.listen(1)

        print("Waiting for Minecraft mod connection...")

        CONN, ADDR = self.TCP_SERVER.accept()
        print("Connected:", ADDR)

        BUFFER = ""

        while True:
            DATA = CONN.recv(self.TCP_BUFFER_SIZE)
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
        frame = cv2.resize(frame, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0

        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)

        return frame

    def get_minecraft_frame(self):
        with mss.MSS() as sct:
            monitor = sct.monitors[self.CAPTURE_MONITOR_INDEX]
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
            self.GAME_CONTROLLER.press_key("w", self.KEY_HOLD_MS)

        elif action == 2:
            self.GAME_CONTROLLER.press_key("a", self.KEY_HOLD_MS)

        elif action == 3:
            self.GAME_CONTROLLER.press_key("d", self.KEY_HOLD_MS)

        elif action == 4:
            self.GAME_CONTROLLER.press_key("s", self.KEY_HOLD_MS)

        elif action == 5:
            self.GAME_CONTROLLER.press_key("space", self.KEY_HOLD_MS)

        elif action == 6:
            self.GAME_CONTROLLER.move_smooth(self.TURN_LEFT_DELTA_X, 0, self.MOUSE_MOVE_MS, self.STEP_DELAY_S)

        elif action == 7:
            self.GAME_CONTROLLER.move_smooth(self.TURN_RIGHT_DELTA_X, 0, self.MOUSE_MOVE_MS, self.STEP_DELAY_S)

        return reward, current_state

    def get_reward(self, prev, curr):
        reward = 0.0

        if curr.position[0] != prev.position[0] or curr.position[2] != prev.position[2]:
            reward += (abs(curr.position[2] - prev.position[2]) + abs(curr.position[0] - prev.position[0])) * self.REWARD_MOVEMENT_PER_BLOCK

        reward -= self.REWARD_DAMAGE_PER_HEART * (prev.health - curr.health)

        # idle penalty ( jump is ignored since it can because the AI to jump and not move )
        if curr.position[0] == prev.position[0] and curr.position[2] == prev.position[2]:
            reward -= self.REWARD_IDLE_PENALTY

        if curr.position[1] > prev.position[1]:
            pass

        reward += self.REWARD_PASSIVE

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
            self.GAME_CONTROLLER.press_key("tab", self.KEY_HOLD_MS)
            self.GAME_CONTROLLER.press_key("enter", self.KEY_HOLD_MS)
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

        self.GAME_CONTROLLER.press_key("t", self.KEY_HOLD_MS)
        self.GAME_CONTROLLER.press_key("esc", self.KEY_HOLD_MS)
        self.GAME_CONTROLLER.press_key("t", self.KEY_HOLD_MS)
        self.GAME_CONTROLLER.press_key("up", self.KEY_HOLD_MS)
        self.GAME_CONTROLLER.press_key("enter", self.KEY_HOLD_MS)

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
            
            self.REWARD_BASELINE = self.DISCOUNT_FACTOR * self.REWARD_BASELINE + (1 - self.DISCOUNT_FACTOR) * REWARD
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
            
            if len(self.MEMORY) == self.BATCH_SIZE:
                loss = 0

                for log_prob, advantage in self.MEMORY:
                    loss += -log_prob * advantage

                self.OPTIMIZER.zero_grad()
                loss.backward()
                self.OPTIMIZER.step()

                self.MEMORY.clear()

            # 7. RESET EPISODE
            if self.episode_done(CURRENT_STATE) or self.STEPS >= self.MAX_STEPS:
                self.end_episode(
                    reason="death" if CURRENT_STATE.health == 0 else "timeout"
                )
                self.HIDDEN = ( 
                    torch.zeros( 1, 1, self.LSTM_HIDDEN, device = self.DEVICE ), 
                    torch.zeros(1, 1, self.LSTM_HIDDEN, device = self.DEVICE )
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
            torch.zeros( 1, 1, PhiAgent.LSTM_HIDDEN, device = PhiAgent.DEVICE ), 
            torch.zeros( 1, 1, PhiAgent.LSTM_HIDDEN, device = PhiAgent.DEVICE )
        )

        PhiAgent.save_checkpoint(PhiAgent.AGENT, PhiAgent.OPTIMIZER, PhiAgent.STEPS, HIDDEN)
        
        PhiAgent.GAME_CONTROLLER.press_key("esc", PhiAgent.KEY_HOLD_MS)
        
        PhiAgent.RUNNING = False
        
        if PhiAgent.TCP_SERVER is not None:
            try:
                PhiAgent.TCP_SERVER.close()
            except Exception:
                pass
            
        try:
            PhiAgent.TCP_SERVER_THREAD.join( timeout=PhiAgent.SHUTDOWN_TIMEOUT_S )
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