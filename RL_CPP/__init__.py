from gymnasium.envs.registration import register

register(
    id="FieldEnv-v0",
    entry_point="RL_CPP.envs:FieldEnv",
)