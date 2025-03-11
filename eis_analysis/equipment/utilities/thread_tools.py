
import threading
import time
import sys

class SThread(threading.Thread):
    def __init__(self, kill_old=True, **kwargs):
        super().__init__(**kwargs)
        self._stop_event = threading.Event()
        
        if kill_old:
            for thread in threading.enumerate():
                if thread.name == self.name:
                    thread.stop()
    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            print(f"Exception in thread {self.name}: {e}", file=sys.stderr)
            raise e
    # @property
    def thread_killed(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()

class DeviceTrigger(threading.Thread):
    def __init__(self, check_callable, trigger_callable, true_count=1, wait_time=1.0, name="DevicePoller"):
        super().__init__(name=name)
        self.check_callable = check_callable
        self.trigger_callable = trigger_callable
        self.true_count = true_count
        self.wait_time = wait_time
        self._stop_event = threading.Event()
        self.previous_state = False
        
        # if kill_old:
        for thread in threading.enumerate():
            if thread.name == self.name:
                thread.stop()


    def run(self):
        true_counter = 0
        while not self._stop_event.is_set():
            current_state = self.check_callable()
            if current_state and not self.previous_state:
                true_counter += 1
                if true_counter >= self.true_count:
                    self.trigger_callable()
                    true_counter = 0
            self.previous_state = current_state
            time.sleep(self.wait_time)

    def stop(self):
        self._stop_event.set()


if __name__ == "__main__":
    # from datetime import datetime
    # from .temp_logger import Incrementor
    # incrementor = Incrementor()
    

    
    # def check_condition():
    #     value = mfia_base.device["demods/0/sample"]()["frequency"].item()
    #     return value >= 4.1e6  # Example condition

    # def trigger_action():
    #     t_stable = incrementor._to_seconds(incrementor.time_stable)
    #     if mfia_base.device.imps[0].enable() == 1:
    #         mfia_base.device.imps[0].enable(0)
    #         # if stable for more than 5 min then this isnt a case of bad stepping
    #         if t_stable > 5*60:
    #             incrementor.next()
    #     # if a bad step, while loop would just move on, otherwise will continue after stable for 5 min
    #     n=0
    #     while incrementor._to_seconds(incrementor.time_stable) < 5*60 and n < 30:
    #         n+=1
    #         time.sleep(1)
    #     mfia_base.device.imps[0].enable(1)
    #     mfia_base.device.imps[0].oneperiod(1)
    #     print(f"Condition met @ {datetime.now().strftime('%y%m%d %H%M%S')}")
    
    # # poller = DevicePoller(mfia_base.device, "demods/0/sample", "frequency", check_condition, trigger_action, true_count=1, wait_time=2.0)
    # poller = DevicePoller(check_condition, trigger_action, true_count=1, wait_time=2.0)
    # poller.start()

    def check_condition(value):
        return value < 1e0  # Example condition

    def trigger_action():
        print("Condition met!")

    # poller = DevicePoller(check_condition, trigger_action, true_count=3, wait_time=2.0)
    # poller.start()

    # # Run for a while then stop
    # time.sleep(60)
    # poller.stop()
    # poller.join()