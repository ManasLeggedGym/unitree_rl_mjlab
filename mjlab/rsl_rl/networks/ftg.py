#implements FGT in : Learning Quadrupedal Locomotion over Challenging Terrain
# Given in the supplimentary section
# convert it to the correct frame for correct angles from aik
import numpy as np
import random 
import matplotlib.pyplot as plt

class FGT:
    def __init__(self,height,f0 = 1.25):
        self.h = height
        self.phi = np.random.uniform(0, 2*np.pi)
        self.f0 = f0 # base frequency set to 1.25 Hz
    
    def get_k(self, phi):
        return 2*(phi-np.pi)/np.pi  

    def update_phi(self,f1,step_time = 0.02):
        w = 2*np.pi*(self.f0+f1)
        self.phi += (w*step_time) 
        return self.phi% (2*np.pi)
    
    def piece_wise_spline(self, k):
        if 0 <= k <= 1:
            z = self.h*(-2*k**3 + 3*k**2) - 0.5
        elif 1 <= k <= 2:
            z = self.h*(2*k**3 - 9*k**2 + 12*k - 4) - 0.5
        else:
            z = -0.5

        return z
    
def main():
    ftg = FGT(height=0.2)

    steps = 400
    dt = 0.02

    time_data = []
    height_data = []
    phase_data = []

    for i in range(steps):

        phi = ftg.update_phi(1.0)
        k = ftg.get_k(phi)

        z = ftg.piece_wise_spline(k)

        time_data.append(i*dt)
        height_data.append(z)
        phase_data.append(phi)

    # -------- Plot 1: Foot height vs time --------
    plt.figure(figsize=(10,4))
    plt.plot(time_data, height_data)
    plt.title("Foot Height vs Time (FTG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Foot Height")
    plt.grid()

    # -------- Plot 2: Foot height vs phase --------
    plt.figure(figsize=(6,4))
    plt.scatter(phase_data, height_data, s=10)
    plt.title("Foot Height vs Phase")
    plt.xlabel("Phase (rad)")
    plt.ylabel("Foot Height")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()



    
