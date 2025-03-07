class ModelAPIConfig:
    def __init__(self, name, endpoint, key):
        self.name = name
        self.endpoint = endpoint 
        self.key = key

    def __str__(self):
        return f"Model API Config:\n Name: {self.name}\n Endpoint: {self.endpoint}"
    


class EvalConfig:
    class AnswerType:
        def __init__(self, MCQ=False, FRQ=False):
            self.MCQ = MCQ
            self.FRQ = FRQ

        def __str__(self):
            return f"MCQ: {self.MCQ}, FRQ: {self.FRQ}"
        
                
    class ConversationType:
        def __init__(self, vignette=False, multi_turn=False, single_turn=False, summarized=False):
            self.vignette = vignette
            self.multiturn = multi_turn
            self.singleturn = single_turn
            self.summarized = summarized
        
        def __str__(self):
            return f"vignette: {self.vignette}, multi_turn: {self.multi_turn}, single_turn: {self.single_turn}, summarized: {self.summarized}"
                
                
    def __init__(self, answer_type_config=None, conversation_type_config=None):
        if answer_type_config is None:
            answer_type_config = {}
        if conversation_type_config is None:
            conversation_type_config = {}
            
        self.answer_type = self.AnswerType(**answer_type_config)
        self.conversation_type = self.ConversationType(**conversation_type_config)

    def __str__(self):
        return f"EvalConfig:\n  Answer Type: {self.answer_type}\n  Conversation Type: {self.conversation_type}"