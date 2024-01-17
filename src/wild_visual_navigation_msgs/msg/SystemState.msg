# Pipeline operation mode
std_msgs/Header header
string mode
# Current valid samples
uint32 main_graph_num_valid_node
# Training loss
float32 total_loss
# Training loss
float32 loss_pred
# Training loss
float32 loss_reco

float32 confidence
# Training step
uint32 step
# Pause learning flag
bool pause_learning
