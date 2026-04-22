from agentshim.base import CodingAgent
from agentshim.trajectory import NullTrajectoryRecorder


class _ConcreteAgent(CodingAgent):
    def generate(self, prompt, cwd=None, timeout=300, silent=False):
        return ""


def test_recorder_default_without_assignment():
    agent = _ConcreteAgent()
    assert isinstance(agent.recorder, NullTrajectoryRecorder)
