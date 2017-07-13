from .config import unknown_person_label


class Smoother(object):

    def __init__(self, model):
        self.model = model
        self.initial_threshold = 3
        self.forfeit_threshold = 100
        self.current_value = 0
        self.candidate = None

    def detect(self, persons):
        if self.model.current is None:
            if self.candidate is None and len(persons) > 0:
                self.candidate = persons[0] if persons[0] != unknown_person_label else None
                self.current_value = 1
            elif self.candidate in persons:
                self.current_value += 1
                if self.current_value >= self.initial_threshold:
                    self.model.current = self.candidate
                    self.current_value = self.forfeit_threshold
            else:
                self.current_value -= 1
                if self.current_value < 0:
                    self.candidate = None
                    self.current_value = 0

        elif self.model.current not in persons:
            if len(persons) > 0:
                self.current_value -= 20
            else:
                self.current_value -= 5
            if self.current_value <= 0:
                self.model.current = None
                self.current_value = 0
        else:
            self.current_value = self.forfeit_threshold

        #print(self.model.current, self.current_value, self.candidate)



