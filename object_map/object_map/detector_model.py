import numpy as np

class DetectorModel:
    def __init__(self):

        self.class_name_by_class_id = {0: "book",
                                        1: "crystalline",
                                        2: "fabric_softner",
                                        3: "sun_screen"}
        self.default_class_name = "unknown"

        self.number_of_classes = 4
        self.asymmetry_by_class_id = {0: True,  # book
                                    1: False,  # crystalline
                                    2: True,  # fabric_softner
                                    3: False}  # sun_screen
        
        self.recall_by_class_id = {0: 0.85,  # book
                                    1: 0.55,  # crystalline
                                    2: 0.70,  # fabric_softner
                                    3: 0.75,}  # sun_screen
        self.default_recall = 0.7

        # P(detection | not exists) per class: false positive prob. (TODo:set from YOLO test + justify)
        self.fp_rate_by_class_id = {0: 0.15,  # book
                                    1: 0.15,  # crystalline
                                    2: 0.15,  # fabric_softner
                                    3: 0.15}  # sun_screen
        self.default_fp_rate = 0.15
        
        self.dirichlet_prior = 1.0

        # Recall vector indexed by class_id, used in update_missed
        self.recall_vec = np.zeros(self.number_of_classes, dtype=float)
        for k in self.recall_by_class_id:
            self.recall_vec[k] = self.recall_by_class_id[k]

        # Precision vector indexed by class_id, used in update_missed
        self.average_fp_rate = float(np.mean(list(self.fp_rate_by_class_id.values())))


        # LiDAR geometric likelihoods (sensor-independent of class)
        self.lidar_p_free_if_exists = 0.15
        self.lidar_p_free_if_not_exists = 0.95
        
        

    def get_recall(self, class_id):
        return self.recall_by_class_id.get(int(class_id), self.default_recall)
    
    def get_fp_rate(self, class_id):
        return self.fp_rate_by_class_id.get(int(class_id), self.default_fp_rate)
    
    def get_asymmetry_by_class_id(self, class_id):
        return self.asymmetry_by_class_id.get(int(class_id), False) # return value from dict, else False (no yaw estimation)
    
    def get_class_name_by_class_id(self, class_id):
        return self.class_name_by_class_id.get(int(class_id), self.default_class_name)