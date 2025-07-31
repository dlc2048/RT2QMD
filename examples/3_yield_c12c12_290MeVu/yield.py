import numpy as np


xs = 8.04008e-23 * 1e22 * 1e3  # barn [mm2] -> mb [cm2], from Glauber-Gribov model


class DDX:
    def __init__(self, file_name: str, nskip: int=0):
        self.data = None

        data = []
        n    = 0
        with open(file_name) as file:  
            for line in file:
                if n < nskip:
                    n += 1
                    continue
                items = line.split()
                data += [list(map(float, items[:4]))]
        self.data = np.array(data)


isotopes = ['b11', 'b10', 'be10', 'be9', 'be7', 'li8', 'li7', 'li6']

for isotope in isotopes:
    tally = DDX('yield_c12c12_{}_0.txt'.format(isotope), nskip=6)
    yld   = tally.data[0,2]
    print('{:<4} Yield = {:.6} mb'.format(isotope, yld * xs * 4 * np.pi))  # account 1/sr
