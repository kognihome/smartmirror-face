from .config import unknown_person_label


class Smoother(object):

    def __init__(self, model):
        self.model = model
        self.initial_threshold = 5
        self.forfeit_threshold = 100
        self.current_value = 0
        self.candidate = None

    def detect(self, persons):

        # in case no person is present
        if self.model.current is None:
            # if a person has been detected and we have not set the 'candidate'
            # a candidate is a person we will 'detect' soon but we want to be sure
            # it is not just a false positive
            if self.candidate is None and len(persons) > 0:
                self.candidate = persons[0] if persons[0] != unknown_person_label else None
                self.current_value = 1
            # we already have a canidate and check if the person has been detected (again)
            elif self.candidate in persons:
                self.current_value += 1
                # the above set threshold defines when we assume that the canidate is actually in front of
                # the mirror
                if self.current_value >= self.initial_threshold:
                    self.model.current = self.candidate
                    # set the current_value to the highest 'heat'
                    self.current_value = self.forfeit_threshold
            else:
                # if no person, or our candidate has not been tracked, reduced the current value
                self.current_value -= 1
                if self.current_value < 0:
                    self.candidate = None
                    self.current_value = 0

        # in case a person is set as 'present' but could not be tracked
        elif self.model.current not in persons:
            # no one is in front of the mirror, reduce the 'heat' faster
            if len(persons) > 0:
                self.current_value -= 20
            else:
                # maybe just a false positive, reduce the current value slightly
                self.current_value -= 5
            # if the value drops below zero, we assume the person is gone
            if self.current_value <= 0:
                self.model.current = None
                self.current_value = 0
        # person there and also detected, keep the value up
        else:
            self.current_value = self.forfeit_threshold



