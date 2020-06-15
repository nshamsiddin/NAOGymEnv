from gym.envs.registration import register

register(id='gym-nao-v0', 
	entry_point='gym_nao.envs:gymNaoEnv', 
)