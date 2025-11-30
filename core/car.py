
class Car:
    def __init__(self, idx, params):
        self.idx = idx
        self.params = params
        self.adc_error_history = [0]
        self.avc_error_history = [0]
        self.time_adc = [0]
        self.time_avc = [0]
        self.avc_error = 0
        self.adc_error = 0
        self.sum_error_Avc = 0
        self.sum_error_Adc = 0
        self.pid_avc = 0
        self.pid_adc = 0
        self.vset = 25
        self.xego = 0
        self.vego = 22
        self.aego = 0
        self.pid = 0
        self.ff = 0
        self.time = [0]
        self.fitness = 10000
        self.step_response_result = {
            'riseTime': [],
            'settlingTime': [],
            'overshoot': [],
            'overshootPercentage': []
        }

    def _to_dict(self):
        return {
            'idx': self.idx,
            'params': self.params,
            'fitness': self.fitness,
            'step_response_result': self.step_response_result
        }