class Configuration(object):
    def __init__(self):
        self.data_path = '/workspace/DBs/CC/ProcessedData/WorldExpo'
        self.dataloader_worker = 8
        self.height = 576
        self.width = 768
        self.aug_degrees = (-20, 20)
        self.aug_translate = (.2, .2)
        self.aug_scale = (0.5, 1.5)
        self.padm_default_distance = 15

        self.test_split_threshold = 100
        self.test_split_depth = 2

config = Configuration()