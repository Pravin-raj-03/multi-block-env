"""
Environment client using the official OpenEnv GenericEnvClient.
"""

from openenv.core.generic_client import GenericEnvClient

class EnvClient:
    """
    A simplified wrapper around OpenEnv's GenericEnvClient.
    Provides a synchronous interface for the trainer.
    """
    def __init__(self, base_url: str):
        self.client = GenericEnvClient(base_url=base_url).sync()
    
    def reset(self, **kwargs):
        """Reset the episode on the remote server."""
        result = self.client.reset(**kwargs)
        # Compatibility: return observation dict
        return result.observation
    
    def step(self, action_text: str):
        """Execute a step on the remote server."""
        action = {"text": action_text}
        result = self.client.step(action)
        # Compatibility: return (obs, reward, done, info)
        return result.observation, result.reward, result.done, {}

    def close(self):
        self.client.close()

