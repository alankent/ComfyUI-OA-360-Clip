from .oa_360_clip import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    WEB_DIRECTORY,
    setup_api_routes,
    populate_cache_from_workflow,
    _node_extra_cache,
)

# Set up API routes when server is available
# Hook into server initialization
def setup():
    """Called by ComfyUI when setting up custom nodes"""
    try:
        import server
        # Wait for server to be initialized
        if hasattr(server, 'PromptServer'):
            # Use a callback to set up routes when server is ready
            original_init = server.PromptServer.__init__
            def patched_init(self, loop):
                original_init(self, loop)
                setup_api_routes(self)
                # Hook into prompt execution to populate cache from workflow
                def on_prompt_handler(json_data):
                    if "prompt" in json_data:
                        prompt = json_data["prompt"]
                        # Populate cache from workflow's node extra data
                        populate_cache_from_workflow(prompt)
                self.add_on_prompt_handler(on_prompt_handler)
            server.PromptServer.__init__ = patched_init
            # If instance already exists, set up routes now
            if hasattr(server.PromptServer, 'instance') and server.PromptServer.instance:
                setup_api_routes(server.PromptServer.instance)
    except Exception as e:
        print(f"[OA360Clip] Warning: Could not set up API routes: {e}")

# Call setup
setup()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

