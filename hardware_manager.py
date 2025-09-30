#%%
import pyvisa
from qolab.hardware.scope.sds800xhd import SDS800XHD
from qolab.hardware.multimeter.hp3457a import HP3457A
from qolab.hardware.multimeter.bk_5491 import BK_5491
from qolab.hardware.multimeter.hp_34401 import HP_34401
from qolab.hardware.vacuum_gauge.mks390 import MKS390
from qolab.hardware.power_supply.psw25045 import PSW25045
from qolab.hardware.power_supply.gpp3610h import GPP3610H
from qolab.hardware.power_supply.keysight_e36231a import KeysightE36231A

#%%
class HardwareManager():
    '''
    Handles automatic initialization of 
    the harware connected throug NI Visa. 
    Collects data from DMMs and PSUs.
    '''
    def __init__(self, visa_addr_dict: dict|None) -> None:
        self.visa_addr_dict = visa_addr_dict
        #----------------------------------------------
        # Open visa resources and initialize intruments
        # scope2 = SDS800XHD(
        #     rm.open_resource(address_dict['scope2'])
        # )
        
        ALLOWED_EQUIPMENT = ['SDS800XHD',
                             'HP3457A',
                             'BK_5491', 
                             'HP_34401', 
                             'MKS390', 
                             'PSW25045', 
                             'GPP3610H',
                             'KeysightE36231A']
        self.rm = pyvisa.ResourceManager()
        self.visa_resources_dict = {}
        for key, (qolab_class, addr) in visa_addr_dict.items():
            if qolab_class in ALLOWED_EQUIPMENT:
                print(qolab_class)
                exec_str = f'{qolab_class}(self.rm.open_resource(\"{addr}\"))'
                self.visa_resources_dict[key] = eval(exec_str)


    def get_resources(self):
        return self.visa_resources_dict
    
    def get_readings(self):
        readings_dict = {}
        for key, visa_resource in self.visa_resources_dict.items():
            if hasattr(visa_resource, 'getAdc'):
                value = visa_resource.getAdc()
                readings_dict[key] = value
            elif hasattr(visa_resource, 'get_pressure'):
                value = visa_resource.get_pressure()
                readings_dict[key] = value
            elif hasattr(visa_resource, 'get_out_current') and hasattr(visa_resource, 'get_out_voltage'):
                value_voltage = visa_resource.get_out_voltage()
                value_current = visa_resource.get_out_current()
                readings_dict[key+' Current (A)'] =  value_current
                readings_dict[key+' Voltage (V)'] =  value_voltage
        return readings_dict
    
    def get_md_table_row(self, header=False):
        readings_dict = self.get_readings()
        readings_str_list = [str(v) for (v,_) in readings_dict.values()]
        data_row_str = "| " + " | ".join(readings_str_list) + " |"
        if header:
            header_str = "| " + " | ".join(readings_dict.keys()) + " |\n"
            separator_str = "|------"*len(readings_dict) + "|\n"
            return header_str + separator_str + data_row_str
        return data_row_str
        


#%%
if __name__ == '__main__':

    addr_dict = {
        'scope1': ('SDS800XHD', 'TCPIP0::192.168.110.191::inst0::INSTR'),
        'scope2': ('SDS800XHD', 'TCPIP0::192.168.110.198::inst0::INSTR'),
        'Grid Current': ('HP3457A', 'visa://192.168.194.15/GPIB1::22::INSTR'),
        'Anode Current': ('HP_34401', 'visa://192.168.194.15/ASRL12::INSTR'),
        'Chamber Current': ('BK_5491', 'visa://192.168.194.15/ASRL9::INSTR'),
        'Pressure': ('MKS390', 'visa://192.168.194.15/ASRL13::INSTR'),
        'Anode Source': ('PSW25045', 'visa://192.168.194.15/ASRL10::INSTR'),
        'Heating Source': ('KeysightE36231A', 'visa://192.168.194.15/USB0::0x2A8D::0x2F02::MY61003701::INSTR'),
    }
    setup = VisaHardware(addr_dict)
#%%
    print(setup.get_md_table_row(True))
# %%
