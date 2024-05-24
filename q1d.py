import numpy as np

class Q1d:
    """
    Class representing steady, quasi-one-dimensional, internal compressible flow with area change and heat addition.
    
    Method derived from "Steady, quasi-one-dimensional, internal compressible flow with area change, heat addition and friction"
    by Andrew A. Oliva and Scott C. Morris

    Methods:
    - __init__(self, N, Tt, A): Initializes the Q1d object.
    - compute_solution(self, M1=0.1, tol=0.05): Computes the solution for the nozzle flow.
    - solve2ndorder(self, M1, AR, Ttr, tol): Solves the 2nd order equation.

    """
     

    def __init__(self, N, Tt, A, gamma=1.4, R=287):
        """
        Initializes the 1D Nozzle object.

        Parameters:
        - N (int): Number of points
        - Tt (list): List of length N containing the total temperature at each point
        - A (list): List of length N containing the cross-sectional area at each point
        - gamma (float, optional): Specific heat ratio (default is 1.4)
        - R (float, optional): Specific gas constant (default is 287)
        """
        self.N = N
        assert len(Tt) == N, "Tt must be of length N"
        assert len(A) == N, "A must be of length N"
        self.As = A
        self.Tt = Tt
        self.gamma = gamma
        self.R = R
    
    def compute_solution(self, M1=0.1, tol=0.05):
        """
        Computes the solution for the nozzle flow.

        Parameters:
        - M1 (float): The initial Mach number. Default is 0.1.
        - tol (float): The absolute tolerance for sonic point transition. Default is 0.03.

        Returns:
        - Ms (numpy.ndarray): An array of Mach numbers representing the solution.

        """
        N_elm = self.N - 1
        # Area ratio
        Ars = self.As[1:] / self.As[:-1]
        # Mach numbers
        Ms = np.zeros(self.N)
        Ms[0] = M1

        # Main loop
        for i in range(N_elm):
            Ms[i + 1] = self.solve2ndorder(Ms[i], self.As[i + 1] / self.As[i], self.Tt[i + 1] / self.Tt[i], tol=tol)
        return Ms

    def solve2ndorder(self, M1, AR, Ttr, tol):
        """
            Solve the 2nd order equation from the given parameters:
            - M1 (float): Inlet Mach number
            - AR (float): Area ratio
            - Ttr (float): Total temperature ratio
            - tol (float): Tolerance for the transonic region
        """
        gamma = self.gamma
        F = 1 + (self.gamma - 1) / 2 * M1 ** 2
        Lambda = ((1 + gamma*M1**2 + (0.5)*(AR-1) ) /(M1*np.sqrt(F*Ttr)))**2 

        # Quadratic coefficients
        b =   (Lambda-2*gamma* (1 + 0.5 * (1/AR-1)))/ \
              (Lambda*(gamma-1)/2 - gamma**2)
        c = - (1 + 0.5 * (1/AR-1))**2 /\
              (Lambda*(gamma-1)/2 - gamma**2)
        
        # Solve the quadratic equation
        discriminant = b**2 - 4*c

        if discriminant < 0:
            raise ValueError("No real solutions, the flow is likely to be thermally choked. Try reducing the inlet Mach number.")
        elif b > 0:
            assert np.abs(b/2) < np.sqrt((b/2)**2 - c), "b/2 must be smaller than sqrt((b/2)**2 - c)"
            M2 = np.sqrt(-b/2 + np.sqrt((b/2)**2 - c))
        elif b < 0 and np.abs(b/2) < np.sqrt((b/2)**2 - c):
            M2 = np.sqrt(-b/2 + np.sqrt((b/2)**2 - c))
        elif b < 0 and np.abs(b/2) > np.sqrt((b/2)**2 - c):
            # Different cases depending on the flow parameters. The method should be free of the Mach number
            # oscilating back and forth between subsonic and supersonic. Because around the sonic point,
            # dM/dTt towards sonic conditions is going to be lower in magnitude than dM/dTt going away from sonic conditions.
              
            # 1. Transonic flow: M1 lower but close to 1 and AR > 1  -> M2 > 1
            if M1 < 1  and np.abs(M1-1)<tol and AR > 1:
                M2 = np.sqrt(-b/2 + np.sqrt((b/2)**2 - c))
            # 2. Supersonic flow: M1 higher but close to 1 and AR > 1  -> M2 < 1
            elif M1 > 1  and np.abs(M1-1)<tol and AR > 1:
                M2 = np.sqrt(-b/2 + np.sqrt((b/2)**2 - c))
            # 3. Supersonic flow but not close to 1  -> M2 > 1
            elif M1 > 1 :
                M2 = np.sqrt(-b/2 + np.sqrt((b/2)**2 - c))
            # 4. Subsonic flow but not close to 1  -> M2 < 1
            else:
                M2 = np.sqrt(-b/2 - np.sqrt((b/2)**2 - c))


        return M2




if __name__ == "__main__":
    N = 100
    Tt = np.linspace(100, 100, N)
    A = np.concatenate((np.linspace(2, 1, N//2), np.linspace(1, 2, N//2)))
    q1d = Q1d(N, Tt, A)
    M1 = 0.4
    while True:
        try:
            Ms = q1d.compute_solution(M1=M1)
            break
        except ValueError as e:
            M1-=0.0001
            print(M1)
            continue
    print(Ms)
    print("Done")



    