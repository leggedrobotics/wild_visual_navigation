from wild_visual_navigation.data_preprocessing import extractor_register

# 1. Verify that in folder all rosbages are present.
# 2. Check if same timestamp is available.


ext = extractor_register["image"]( bagdir = "/media/Data/Datasets/Perugia/day3/mission_data/2022-05-12T11:44:56_mission_0_day_2", outdir = "/media/Data/Datasets/Perugia/preprocessing_test",  tag = 0)
ext.update()
