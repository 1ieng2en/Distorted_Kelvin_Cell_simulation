import numpy as np
from beef import fe
from beef.modal import normalize_phi
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.fftpack import fft


class kc_simulate:
    def __init__(self, kc):
        self.kc = kc
        self.kc_section_params = dict(
            A=0.0625e-6*np.pi,          # Cross-sectional area (m^2)
            m = 160,           # Mass per unit length (kg/m)
            I_z=1e-8,        # Second moment of area about z-axis (m^4)
            I_y=1e-8,        # Second moment of area about y-axis (m^4)
            E=563e6,          # Young's modulus (Pa)
            J=1e-8,          # Torsional constant (m^4)
            poisson=0.3      # Poisson's ratio
        )
    
    def build_assembly(self, no_fix = False):
        kc_section = fe.Section(**self.kc_section_params, name='Kelvin Cell section')

        # Use kc.vertices as the node matrix
        node_labels = np.arange(1, self.kc.vertices.shape[0] + 1).astype(int)
        node_matrix = np.hstack((node_labels.reshape(-1, 1), self.kc.vertices))

        # Use kc.connectivity as the element matrix, converting to 1-based indexing
        element_labels = np.arange(1, len(self.kc.connectivity) + 1).astype(int)
        element_matrix = np.hstack((element_labels.reshape(-1, 1), np.array(self.kc.connectivity) + 1))

        # Define the part using the node and element matrices
        part = fe.Part(node_matrix, element_matrix, [kc_section] * element_matrix.shape[0])

        # 定义固定第一平面的节点
        fix_nodes = [17, 18, 19, 20]
        if no_fix:
            fix_nodes = [17]
        constraints_fix = [fe.Constraint(fix_nodes, dofs='all', node_type='beam3d')]

        # 获取所有节点标签
        all_nodes = set(node_labels.tolist())
        
        # 找到共享节点
        shared_nodes = list(all_nodes - set(fix_nodes))  # 这里移除固定的节点，剩下的就是共享节点

        # 定义弹性约束的刚度
        E = self.kc_section_params['E']
        k_stiff = E  # 刚性较大的弹簧
        k_flexible = E * 0.01  # 刚性较小的弹簧（例如，10% 的 E）

        # 为这个共享节点定义弹性约束
        # 为这些共享节点定义弹簧连接
        spring_connections = []
        for node in shared_nodes:
            spring_connections.append(fe.Spring([node], [0, 1, 2], k_stiff))  # 在平移方向上的较小刚度
            spring_connections.append(fe.Spring([node], [3, 4, 5], k_stiff))  # 在旋转方向上的较大刚度

        # 合并约束
        constraints = constraints_fix

        # 定义组装
        assembly = fe.Assembly([part], constraints=constraints, features=spring_connections)
        
        
        return assembly

    def eigen_mode(self, mode_ix, scaling):
        assembly = self.build_assembly()

        # Perform eigenvalue analysis
        self.analysis = fe.Analysis(assembly, rayleigh = {'stiffness': 1e-4, 'mass': 1e-3})
        self.lambd, self.phi = self.analysis.run_eig()
        self.phi = normalize_phi(self.phi)

        self.analysis.eldef.deform(self.phi[:, mode_ix] * scaling)
        print(f'Mode {mode_ix+1}, f = {np.abs(np.imag(self.lambd[mode_ix])/2/np.pi):.4f} Hz')
        # sc = analysis.eldef.plot(plot_states=['undeformed', 'deformed'])


    def run_dynamic_analysis(self, mode_freq,node_labels, tmax=60, dt=0.1):
        assembly = self.build_assembly()
        rayleigh = {'stiffness': 1e-4, 'mass': 1e-3}

        self.t_force = np.arange(0, tmax, dt)
        om_force = mode_freq * (2 * np.pi)
        self.force_amplitude = 1000 * np.sin(self.t_force * om_force)

        forces = [fe.Force(node_labels, 3, self.force_amplitude, t=self.t_force)]
        
        self.analysis = fe.Analysis(assembly, forces=forces, t0=0, dt=dt, 
                                    tmax=tmax, rayleigh=rayleigh)
        self.analysis.run_lin_dynamic()

        nodes_to_plot = [5, 6, 7, 8]
        self.avg_displacement = np.mean([self.analysis.u[node*6 + 1, :] for node in nodes_to_plot], axis=0)


    def apply_force_to_node(self, tmax, dt, force_node, direction=1, rayleigh=None):
        assembly = self.build_assembly(no_fix=True)

        # Calculate the frequency range using the sampling theorem
        self.n_steps = int(tmax / dt)
        f_max = 1 / (2 * dt)
        f_min = 1 / tmax
        self.freq_range = np.linspace(f_min, f_max, self.n_steps//2)
        
        # Set default Rayleigh damping if not provided
        if rayleigh is None:
            rayleigh = {'stiffness': 0.0, 'mass': 0.0}
        
        # Prepare the force signal for each frequency
        self.force_amplitude = []
        self.analyses = []

        for freq in self.freq_range:
            om = 2 * np.pi * freq
            t = np.arange(0, tmax, dt)
            force_amplitude = 1000 * np.sin(om * t)
            force = [fe.Force(force_node, direction, force_amplitude, t=t)]
            analysis = fe.Analysis(assembly, forces=force, t0=0, dt=dt, tmax=tmax, rayleigh=rayleigh)
            analysis.run_lin_dynamic()
            
            # Store analysis results and force amplitude for later use
            self.analyses.append(analysis)
            self.force_amplitude.append(force_amplitude)

    def fft_mobility(self, disp_node, direction=1):
        mobility_response = []
        diss = []
        for i, analysis in enumerate(self.analyses):
            # Perform FFT on displacement response
            for node in disp_node:
                dis_response = analysis.u[node*6 + direction, :]
                diss.append(dis_response)

            displacement_response = np.mean(diss, axis=0)
            displacement_fft = np.fft.fft(displacement_response)[:self.n_steps//2]
            force_fft = np.fft.fft(self.force_amplitude[i] * 1e6)[:self.n_steps//2]

            # Calculate mobility as the ratio of displacement to force in frequency domain
            mobility = displacement_fft / force_fft
            mobility_response.append(np.abs(mobility))
        
        # Convert to numpy array for easy plotting
        self.mobility_response = np.mean(mobility_response, axis=0)


    def plot_mobility_response(self):
        if self.freq_range is None or self.mobility_response is None:
            raise ValueError("Mobility response has not been calculated. Run run_mobility_response() first.")
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.freq_range, self.mobility_response)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Mobility [m/N]')
        plt.title('Mobility Response')
        plt.grid(True)
        plt.show()







class SeismicSimulation(kc_simulate):
    def __init__(self, model, material_params, seismic_motion):
        self.model = model
        self.material_params = material_params
        self.seismic_motion = seismic_motion  # Function describing base motion

    def build_assembly(self):
        # Define section properties based on the material parameters provided
        section = fe.Section(**self.material_params, name='Model Section')

        # Use model vertices as the node matrix
        node_labels = np.arange(1, self.model.vertices.shape[0] + 1).astype(int)
        node_matrix = np.hstack((node_labels.reshape(-1, 1), self.model.vertices))

        # Use model connectivity as the element matrix, converting to 1-based indexing
        element_labels = np.arange(1, len(self.model.connectivity) + 1).astype(int)
        element_matrix = np.hstack((element_labels.reshape(-1, 1), np.array(self.model.connectivity) + 1))

        # Define the part using the node and element matrices
        part = fe.Part(node_matrix, element_matrix, [section] * element_matrix.shape[0])

        # No fixed constraints, as we are simulating a seismic condition
        constraints = []

        # Define the assembly
        assembly = fe.Assembly([part], constraints=constraints)
        return assembly

    def apply_seismic_boundary(self, tmax, dt):
        assembly = self.build_assembly()
        t = np.arange(0, tmax, dt)
        base_displacement = self.seismic_motion(t)
        
        # Apply the seismic displacement to the bottom nodes
        bottom_nodes = [17, 18, 19, 20]
        # seismic_forces = [fe.Displacement(bottom_nodes, direction=1, displacement=base_displacement, t=t)]
        
        self.analysis = fe.Analysis(assembly, displacements=base_displacement, t0=0, dt=dt, tmax=tmax)
        self.analysis.run_lin_dynamic()

    def measure_response(self):
        top_nodes = [5 , 6, 7, 8]
        bottom_nodes = [17, 18, 19, 20]

        # Measure displacement response at top and bottom
        top_displacement = np.mean([self.analysis.u[node*6 + 1, :] for node in top_nodes], axis=0)
        bottom_displacement = np.mean([self.analysis.u[node*6 + 1, :] for node in bottom_nodes], axis=0)

        return top_displacement - bottom_displacement

    def plot_response(self, response):
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(response)), response)
        plt.xlabel('Time Steps')
        plt.ylabel('Displacement Difference [m]')
        plt.title('Displacement Response Over Time')
        plt.grid(True)
        plt.show()