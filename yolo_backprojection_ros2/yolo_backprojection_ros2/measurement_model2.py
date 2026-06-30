
import numpy as np

class MeasurementModel:
    def __init__(self, class_mapping=None):
        
        if class_mapping is None:
            class_mapping = {0: {"name": "book", "symmetric": False}, 
                             1: {"name": "crystalline", "symmetric": True}, 
                             2: {"name": "fabric_softner", "symmetric": False},
                             3: {"name": "sun_screen", "symmetric": True}}

        
        self.class_mapping = class_mapping

        # Inflation factors, multiply based on per axis NSE values
        self.NSE_inflation = {"default": {"pos_x": 4.5, 
                                          "pos_y": 4.5, 
                                          "pos_z": 3.0, 
                                          "yaw": 2.5, 
                                          "extent_x": 5.0, 
                                          "extent_y": 2.5, 
                                          "extent_z": 5.0}}

        # Minimum covariance, checks and set at if below at runtime
        self.min_cov = {"pos_x": 0.01**2,
                        "pos_y": 0.01**2,
                        "pos_z": 0.02**2,
                        "yaw": 0.01**2,
                        "extent_x": 0.01**2,
                        "extent_y": 0.01**2,
                        "extent_z": 0.01**2}

        self.coefficients = {'book': {'r': {'bias': -0.026935, 'cov': 0.000285},
                                    't': {'bias': -0.02196, 'cov': 6.4e-05},
                                    'z': {'bias': -0.007252, 'cov': 1.5e-05},
                                    'l_x': {'bias': -0.008509, 'cov': 3.1e-05},
                                    'l_y': {'bias': -0.012858, 'cov': 0.000944},
                                    'l_z': {'bias': -0.000865, 'cov': 6e-06},
                                    'yaw': {'bias': 0.116013, 'cov': 0.108074}},
                            'crystalline': {'r': {'bias': -0.03381, 'cov': 1.9e-05},
                                            't': {'bias': -0.020941, 'cov': 3.4e-05},
                                            'z': {'bias': -0.008608, 'cov': 1.4e-05},
                                            'l_x': {'bias': -0.007309, 'cov': 0.000156},
                                            'l_y': {'bias': -0.007309, 'cov': 0.000156},
                                            'l_z': {'bias': -0.028032, 'cov': 8e-05}},
                            'fabric_softner': {'r': {'bias': -0.040728, 'cov': 0.000148},
                                                't': {'bias': -0.022169, 'cov': 0.00013},
                                                'z': {'bias': -0.004252, 'cov': 0.000119},
                                                'l_x': {'bias': -0.045218, 'cov': 4.6e-05},
                                                'l_y': {'bias': -0.02333, 'cov': 0.000122},
                                                'l_z': {'bias': -0.053745, 'cov': 0.000158},
                                                'yaw': {'bias': 0.477724, 'cov': 0.205552}},
                            'sun_screen': {'r': {'bias': -0.026459, 'cov': 4.8e-05},
                                            't': {'bias': -0.020595, 'cov': 6.8e-05},
                                            'z': {'bias': -0.004972, 'cov': 1.1e-05},
                                            'l_x': {'bias': 0.000293, 'cov': 1.7e-05},
                                            'l_y': {'bias': 0.000293, 'cov': 1.7e-05},
                                            'l_z': {'bias': -0.009407, 'cov': 5.5e-05}}}


    
    def get_bias_and_covariance_v2(self, class_id, surviving_fraction=None, object_z_elevation=None):
        # 0. if no coefficients, return zero cov
        if self.coefficients is None:
            pos_bias = np.zeros(3, dtype=float)
            pos_cov = np.zeros((3, 3), dtype=float)

            extent_bias = np.zeros(3, dtype=float)
            extent_cov = np.zeros((3, 3), dtype=float)

            yaw_bias = 0.0
            yaw_var = 0.0

            return pos_bias, pos_cov, extent_bias, extent_cov, yaw_bias, yaw_var
        
        # 1. get coefficients for this class from dict
        class_name = self.class_mapping[class_id]["name"]
        class_coeffs = self.coefficients[class_name]

        pos_bias = np.array([class_coeffs["r"]["bias"],
                            class_coeffs["t"]["bias"],
                            class_coeffs["z"]["bias"]], dtype=float)

        pos_cov = np.diag([class_coeffs["r"]["cov"],
                            class_coeffs["t"]["cov"],
                            class_coeffs["z"]["cov"]])

        extent_bias = np.array([class_coeffs["l_x"]["bias"],
                                class_coeffs["l_y"]["bias"],
                                class_coeffs["l_z"]["bias"]], dtype=float)

        extent_cov = np.diag([class_coeffs["l_x"]["cov"],
                                class_coeffs["l_y"]["cov"],
                                class_coeffs["l_z"]["cov"]])

        if "yaw" in class_coeffs:
            yaw_bias = class_coeffs["yaw"]["bias"]
            yaw_var = class_coeffs["yaw"]["cov"]
        else:
            yaw_bias = 0.0
            yaw_var = 0.0

        
        # 2. Apply inflation, based on NSE values from calibration set
        inflation = self.NSE_inflation["default"]

        pos_cov[0, 0] *= inflation["pos_x"]
        pos_cov[1, 1] *= inflation["pos_y"]
        pos_cov[2, 2] *= inflation["pos_z"]

        extent_cov[0, 0] *= inflation["extent_x"]
        extent_cov[1, 1] *= inflation["extent_y"]
        extent_cov[2, 2] *= inflation["extent_z"]

        yaw_var *= inflation["yaw"]


        # 3. Check and aply minimum covariances
        pos_cov[0, 0] = max(pos_cov[0, 0], self.min_cov["pos_x"])
        pos_cov[1, 1] = max(pos_cov[1, 1], self.min_cov["pos_y"])
        pos_cov[2, 2] = max(pos_cov[2, 2], self.min_cov["pos_z"])

        extent_cov[0, 0] = max(extent_cov[0, 0], self.min_cov["extent_x"])
        extent_cov[1, 1] = max(extent_cov[1, 1], self.min_cov["extent_y"])
        extent_cov[2, 2] = max(extent_cov[2, 2], self.min_cov["extent_z"])

        yaw_var = max(yaw_var, self.min_cov["yaw"])

        


        return pos_bias, pos_cov, extent_bias, extent_cov, yaw_bias, yaw_var
