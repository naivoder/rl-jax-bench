"""Build harness/cleanrl_ppo_lstm_maskedcartpole.py from CleanRL@fe8d8a0 ppo_atari_lstm.py by mechanical substitution:
(1) make_env -> gymnasium CartPole-v1 + RecordEpisodeStatistics + velocity mask (obs[1]=obs[3]=0) [+ optional
    gymnasium NormalizeReward/clip when MASKED_RSCALE=1, CleanRL's continuous-script wrapper pair];
(2) the Nature-CNN encoder -> Linear(4,64)+Tanh (layer_init), nn.LSTM(512,128) -> nn.LSTM(64,128);
(3) the unused cleanrl_utils.atari_wrappers import is dropped; get_states: x / 255.0 -> x. Everything else (rollout, LSTM state handling L1-L5, GAE, env-column minibatches,
    loss, optimiser) is CleanRL's code unchanged. Prints sha256 of input and output."""
import hashlib
import os

H = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))
src = open(os.path.join(H, "cleanrl", "cleanrl", "ppo_atari_lstm.py")).read()
i, j = src.index("def make_env(env_id, idx, capture_video, run_name):"), src.index("def layer_init(")
new_env = '''class _MaskVelocity(gym.ObservationWrapper):
    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        obs[1] = 0.0
        obs[3] = 0.0
        return obs


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = _MaskVelocity(env)
        if os.environ.get("MASKED_RSCALE", "0") == "1":
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


'''
out = src[:i] + new_env + src[j:]
k0 = out.index("from cleanrl_utils.atari_wrappers import (")
k1 = out.index(")", k0)
out = out[:k0] + "# (atari wrapper import removed: unused on CartPole)" + out[k1 + 1:]
a, b = out.index("        self.network = nn.Sequential("), out.index("        self.lstm = nn.LSTM(512, 128)")
out = out[:a] + "        self.network = nn.Sequential(layer_init(nn.Linear(4, 64)), nn.Tanh())\n" + out[b:]
out = out.replace("        self.lstm = nn.LSTM(512, 128)", "        self.lstm = nn.LSTM(64, 128)")
assert "self.network(x / 255.0)" in out
out = out.replace("self.network(x / 255.0)", "self.network(x)")
p = os.path.join(H, "harness", "cleanrl_ppo_lstm_maskedcartpole.py")
open(p, "w").write(out)
print("ppo_atari_lstm.py", hashlib.sha256(src.encode()).hexdigest())
print("cleanrl_ppo_lstm_maskedcartpole.py", hashlib.sha256(out.encode()).hexdigest())
