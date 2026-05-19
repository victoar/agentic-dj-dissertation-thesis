import agentic_dj.agent.tools as tool_module
sp = tool_module._spotify._get_sp()
devices = sp.devices()
print(devices)