from gymnasium.envs.registration import register

register(
    id="FieldEnv-v1",
    entry_point="RL_CPP.envs:FieldEnv",
)