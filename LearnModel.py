import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from dataclasses import dataclass
import pickle
from scipy.optimize import minimize




@dataclass
class PushData:
    alpha_t: float
    beta_t: float
    g: np.ndarray

class DataCollector:
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self._data = data 


    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        cache = self._data[idx]
        alpha_t = cache['alpha_t']
        beta_t = cache['beta_t']
        g = self.get_object_motion(cache['g'])
        return PushData(alpha_t, beta_t, g)
    
    def __iter__(self):
        for cache in self._data:
            alpha_t = cache['alpha_t']
            beta_t = cache['beta_t']
            g = self.get_object_motion(cache['g'])
            yield PushData(alpha_t, beta_t, g)

    @property
    def data(self):
        u_t = np.array([[d.alpha_t, d.beta_t] for d in self])
        g_values = np.array([d.g for d in self])
        return u_t, g_values

    @staticmethod
    def get_object_motion(g):
        tVec = g[:2, -1]
        theta = np.arctan2(g[1, 0], g[0, 0])
        X_t = np.array([tVec[0], tVec[1], theta])
        return X_t


class GPRModel:
    def __init__(self):
        kernel_fwd = C(1.0) * RBF([1.0, 1.0]) + WhiteKernel(noise_level=0.1)
        kernel_inv = C(1.0) * RBF([1.0, 1.0, 1.0]) + WhiteKernel(noise_level=0.1)
        self.gamma = GaussianProcessRegressor(kernel=kernel_fwd, n_restarts_optimizer=5, normalize_y=True)
        self.gamma_inv = GaussianProcessRegressor(kernel=kernel_inv, n_restarts_optimizer=5, normalize_y=True)

    def train(self, D:DataCollector):
        
        if len(D) < 2:
            raise ValueError("Not enough data points to train the model.")
        
        u_t, g_t = D.data
        self.gamma.fit(u_t, g_t)
        self.gamma_inv.fit(g_t, u_t)
    
    def predict(self, b_hat_u_i):
        return self.gamma.predict(b_hat_u_i.reshape(1, -1))[0]
    
    # def predict_inverse(self, b_hat_g_i):
    #     return self.gamma_inv.predict(b_hat_g_i.reshape(1, -1))[0]

    def visualize_compare(self, D:DataCollector):
        u_t, g_t = D.data
        g_pred = self.gamma.predict(u_t)
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(g_t[:, 0], g_t[:, 1], g_t[:, 2], color='blue', label='True')
        ax1.scatter(g_pred[:, 0], g_pred[:, 1], g_pred[:, 2], color='red', label='Predicted')
        ax1.set_title('Forward Model Predictions')
        ax1.set_xlabel('Translation X')
        ax1.set_ylabel('Translation Y')
        ax1.set_zlabel('Rotation Theta')
        ax1.legend()
        plt.show()

    def visualize_gamma(self):
        # Visualize each component of g separately
        alpha_range = np.linspace(0, 2 * np.pi, 100)
        beta_range = np.linspace(-0.2, 0.2, 100)
        alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)
        u_grid = np.column_stack((alpha_grid.ravel(), beta_grid.ravel()))
        g_pred_grid = self.gamma.predict(u_grid).reshape(alpha_grid.shape + (3,))
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ['Translation X', 'Translation Y', 'Rotation Theta']
        for i in range(3):
            ax = axes[i]
            im = ax.contourf(alpha_grid, beta_grid, g_pred_grid[:, :, i], levels=50, cmap='viridis')
            ax.set_title(titles[i])
            ax.set_xlabel('Alpha (radians)')
            ax.set_ylabel('Beta (radians)')
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    def visualize_gamma_inverse(self, theta_fixed=0.0):
        # Visualize the inverse model predictions at a fixed theta
        translation_range = np.linspace(-50, 50, 100)
        tx_grid, ty_grid = np.meshgrid(translation_range, translation_range)
        theta_grid = np.full_like(tx_grid, theta_fixed)
        g_grid = np.column_stack((tx_grid.ravel(), ty_grid.ravel(), theta_grid.ravel()))
        u_pred_grid = self.gamma_inv.predict(g_grid).reshape(tx_grid.shape + (2,))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        titles = ['Alpha (radians)', 'Beta (radians)']
        for i in range(2):
            ax = axes[i]
            im = ax.contourf(tx_grid, ty_grid, u_pred_grid[:, :, i], levels=50, cmap='viridis')
            ax.set_title(titles[i])
            ax.set_xlabel('Translation X')
            ax.set_ylabel('Translation Y')
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'Inverse Model (theta={theta_fixed:.2f})')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path="gpr_model.pickle"):
        with open(path, 'wb') as f:
            pickle.dump((self.gamma, self.gamma_inv), f)

    @classmethod
    def load_model(cls, path="gpr_model.pickle"):
        obj = cls.__new__(cls)
        with open(path, 'rb') as f:
            obj.gamma, obj.gamma_inv = pickle.load(f)
        return obj

    def predict_inverse(self, target_g):
        def cost(u):
            return np.sum((self.predict(u) - target_g)**2)
        result = minimize(cost, x0=np.array([np.pi, 0.0]), method='L-BFGS-B',
                        bounds=[(0, 2*np.pi), (-0.2, 0.2)])
        return result.x
    
if __name__ == "__main__":

    gpr = GPRModel.load_model("gpr_model.pickle")
    # gpr.visualize_gamma()
    gpr.visualize_gamma_inverse()
    # print("Model loaded successfully.")
    data_collector = DataCollector("/home/robot/push_proj2026/pushing_data_1771222247.pickle")
    # test gamma inverse prediction
    test_g = data_collector.data[1][0] # example object motion
    predicted_u = gpr.predict_inverse(test_g)
    print(f"Predicted control for g={test_g}: {predicted_u}")
    # gpr_model = GPRModel()
    # gpr_model.train(data_collector)
    # print("Model trained successfully.")
    # gpr_model.visualize_compare(data_collector)
    # gpr_model.visualize_separate()
    # gpr_model.save_model()
    
