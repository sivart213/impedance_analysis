
from zhinst.toolkit import Session

class MFIABase:
    def __init__(self, server_host: str, device_name: str = "dev6037", interface: str = "1GbE"):
        self.server_host = server_host
        self.session = Session(server_host)
        self.device = self.session.connect_device(device_name, interface=interface)
    
    @property
    def impedance_run_status(self):
        return self.device.imps[0].enable()
    
    @property
    def frequency(self):
        # return self.device["demods/0/sample"]()["frequency"].item()
        return self.async_get("demods/0/freq")
    
    @property
    def auxin0(self):
        # return self.device["demods/0/sample"]()["auxin0"].item()
        return self.async_get("auxins/0/values/0")
    
    @property
    def auxin1(self):
        # return self.device["demods/0/sample"]()["auxin1"].item()
        return  self.async_get("auxins/0/values/1")
    
    def _clean_path(self, path):
        path = str(path)
        path = path.replace(str(self.device[""]), "")
        return path
    
    def get_node_info(self, path, return_info_obj=False):
        info = self.device[self._clean_path(path)].node_info
        if return_info_obj:
            return info
        print(info)
        
    def is_node(self, path, any_valid=True):
        return self.device[self._clean_path(path)].is_valid()

    def child_node_list(self, path, as_nodes=False, recursive=True, leavesonly=True, **kwargs):
        path = self._clean_path(path)
        if self.is_node(path):
            nodes =  list(self.device[path].child_nodes(recursive=recursive, leavesonly=leavesonly, **kwargs))
            if as_nodes:
                return nodes
            return [str(n) for n in nodes]
        
    def async_get(self, path, key="value"):
        path = self._clean_path(path)
        if self.is_node(path):
            self.device[path].get_as_event()
            res = self.session.poll()
            if not res:
                res = self.device[path]()
                if not res:
                    print("failed to get value")
                    return
            if not isinstance(res, dict):
                if hasattr(res, "to_dict"):
                    res = res.to_dict()
                else:
                    return res
            
            key = str(key)
            if key in res:
                return res[key].item() if not isinstance(res[key], (float, int)) and len(res[key]) == 1 else res[key]
            elif any(key in v for v in res.values()):
                kres = []
                for v in res.values():
                    if key in v:
                        vres = v[key].item() if not isinstance(v[key], (float, int)) and len(v[key])  == 1 else v[key]
                        kres.append(vres)
                if len(kres) == 1:
                    kres = kres[0]
                return kres
    

    
    def toggle_imps_module(self):
        if self.device.imps[0].enable() == 1:
            self.device.imps[0].enable(0)

        else:
            self.device.imps[0].enable(1)
            self.device.imps[0].oneperiod(1)
        
        

# Example usage
if __name__ == "__main__":
    # from datetime import datetime
    # from .temp_logger import Incrementor
    # incrementor = Incrementor()
    
    mfia_base = MFIABase("localhost")
    
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