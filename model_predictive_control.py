import numpy as np
import matplotlib.pyplot as plt
import pickle
from LearnModel import GPRModel


class MPC():
    #sigma_t, sigma_theta --> perturbation parameters for the random transformation in line 6 of the algorithm
    def __init__(self, gpr_model:GPRModel, Q=8, L=6, sigma_t=1.0, sigma_theta=0.05):
        """
        gpr_model: trained gpr instance from LearnModel.py
        Q: number of predicted trajectories you want
        L: prediction horizon (steps)
        """
        self.gpr_model = gpr_model
        self.Q = Q
        self.L = L
        self.sigma_t = sigma_t
        self.sigma_theta = sigma_theta
        # self.gpr_model.visualize_gamma()

    # ---- SE(2) helpers ----

    @staticmethod
    def pose_to_SE2(pose):
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, x],
                         [s,  c, y],
                         [0,  0, 1]])
    @staticmethod
    def SE2_to_pose(H):
        return np.array([H[0, 2], H[1, 2], np.arctan2(H[1, 0], H[0, 0])])

    @staticmethod
    def pose_to_SE3(pose, z_offset=0):
        x, y, theta = pose
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0, x],
                         [s,  c, 0, y],
                         [0,  0, 1, z_offset],
                         [0,  0, 0, 1]])
    
    @staticmethod
    def SE3_to_pose(H):
        return np.array([H[0, 3], H[1, 3], np.arctan2(H[1, 0], H[0, 0])])

    #line 6: g_xi = RANDTRANSFORMATION()
    def perturbation_g_xi(self):
        dx = np.random.normal(0, self.sigma_t)
        dy = np.random.normal(0, self.sigma_t)
        dtheta = np.random.normal(0, self.sigma_theta)
        c, s = np.cos(dtheta), np.sin(dtheta)
        return np.array([[c, -s, dx],
                         [s,  c, dy],
                         [0,  0,  1]])

    def trajectory_cost(self, trajectory, Y):
        cost = 0.0
        for X_hat in trajectory:
            X_hat_pose = self.SE2_to_pose(X_hat)
            cost += np.min(np.linalg.norm(Y - X_hat_pose, axis=1))
        return cost


    def visualize_trajectories(self, trajectories, Y, trajectory_costs):
        plt.figure(figsize=(8, 6))
        plt.plot(Y[:, 0], Y[:, 1], 'r-', label='Reference Path')
        plt.scatter(Y[:, 0], Y[:, 1], color='green', label='Waypoints')
        for i, traj in enumerate(trajectories):
            traj_poses = np.array([self.SE2_to_pose(X_hat) for X_hat in traj])
            plt.plot(traj_poses[:, 0], traj_poses[:, 1], label=f'Trajectory {i} (Cost: {trajectory_costs[i]:.2f})')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('MPC Trajectories')
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.show()

    # Algorithm 4: Model Predictive Control
    def run(self, X_t, Y, visualize=False):
        """
        Input:
        X_t:     observed object's configuration
        Y:       reference path
        Output: 
        b_u_t:   generated control for execution
        """
        best_cost = np.inf
        best_u = None
        trajectories = []
        controls = []

        for _ in range(self.Q):
            trajectory = []
            control = []
            # line 2: T_q <- {X_hat_q_0 = X_t}
            X_hat = self.pose_to_SE2(X_t)
            trajectory.append(X_hat)
            # line 3: for k = 0 to L-1 do
            for k in range(self.L):
                X_hat_pose = self.SE2_to_pose(X_hat)

                # line 4: j* = argmin_j ||Y_j - X_hat_q_k||
                distances = np.linalg.norm(Y - X_hat_pose, axis=1)
                j_star = np.argmin(distances)

                # boundary check: j*+1 must be within Y
                if j_star + 1 >= len(Y):
                    j_star = len(Y) - 2

                # line 5: b_g_hat = X_hat^{-1} * Y_{j*+1}
                H_target = self.pose_to_SE2(Y[j_star + 1])
                b_g_hat = np.linalg.inv(X_hat) @ H_target

                # line 6: g_xi = RANDTRANSFORMATION()
                g_xi = self.perturbation_g_xi()

                # line 7: b_u_hat = Γ^{-1}(b_g_hat * g_xi)
                perturbed = b_g_hat @ g_xi
                perturbed_pose = self.SE2_to_pose(perturbed)
                b_u_hat = self.gpr_model.predict_inverse(perturbed_pose)
                control.append(b_u_hat)

                # line 8: X_hat_{k+1} = X_hat * Γ(b_u_hat)
                g_pred = self.gpr_model.predict(b_u_hat)  # [tx, ty, theta]
                X_hat = X_hat @ self.pose_to_SE2(g_pred)

                # line 9: trajectory.append(X_hat_{k+1})
                trajectory.append(X_hat)
                # accumulate cost: min_j ||Y_j - X_hat_{k+1}||
                # X_hat_next_pose = self.SE2_to_pose(X_hat_next)
            # # line 11: select q* with minimum cost
            # if trajectory_cost < best_cost:
            #     best_cost = trajectory_cost
            #     best_u = first_control
            trajectories.append(trajectory)
            controls.append(control)
        # line 12: q* = argmin_q((min_j ||Y_j - X_hat||))
        costs = [self.trajectory_cost(traj, Y) for traj in trajectories]
        # line 13: b_u_t = b_u_hat_{q*,0}
        q_star = np.argmin(costs)
        best_u = controls[q_star][0]
        # line 14: return b_u_t
        if visualize:
            self.visualize_trajectories(trajectories, Y, costs)
        return best_u


if __name__ == "__main__":
    #load trained model
    model_path = "gpr_model.pickle"
    gpr_model = GPRModel.load_model(model_path)
    #instance of MPC
    Q=8
    L=6
    sigma_t=5.0 #millimeters
    sigma_theta=0.3 #radians
    mpc = MPC(gpr_model, Q=Q, L=L, sigma_t=sigma_t, sigma_theta=sigma_theta)
    # #generate a linear reference path for testing
    # Y = np.array([[i, 0, 0] for i in range(0, 500, 50)]).reshape(-1, 3) # straight line path along x-axis
    #generate a curved reference path for testing
    t = np.linspace(0, 2 * np.pi, 18)
    Y = np.column_stack((150 * np.cos(t), 150 * np.sin(t), np.zeros_like(t))) # circular path with radius 100mm
    #visualize the reference path
    # plt.plot(Y[:, 0], Y[:, 1], 'r-', label='Reference Path')
    # plt.scatter(Y[:, 0], Y[:, 1], color='green', label='Waypoints')
    # plt.xlabel('X (mm)')
    # plt.ylabel('Y (mm)')
    # plt.title('Reference Path for MPC')
    # plt.legend()
    # plt.axis('equal')
    # plt.grid()
    # plt.show()
    #test the MPC with an initial object pose
    X_t = np.array([0, -10, 0]) # 10mm below the first waypoint
    b_u_t = mpc.run(X_t, Y, visualize=True)
    print(f"Generated control (alpha, beta): {b_u_t}")