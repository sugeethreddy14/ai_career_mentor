# Stores or manages session/user-specific info
class Memory:
    def __init__(self):
        self.sessions = {}  # Placeholder

    def get_session(self, user_id):
        return self.sessions.get(user_id, {})
