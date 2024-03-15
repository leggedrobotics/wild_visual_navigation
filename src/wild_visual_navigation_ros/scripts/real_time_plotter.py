import pyqtgraph as pg
# import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets,QtCore
import random
import torch
import threading
from PyQt5.QtCore import QTimer

def generate_ticks(start, end, spacing):
    """Generate tick locations and labels for a given range and spacing."""
    ticks = []
    val = start
    while val <= end:
        ticks.append(val)
        val += spacing
    # Return as a list of (location, label) pairs
    return [(tick, str(tick)) for tick in ticks]

def generate_mock_obs(step, episode):
    """Generate a mock observation for a given step and episode."""
    fric = torch.tensor([[episode * 0.1 + (step % 10) * 0.01 for _ in range(4)]])
    stiff = torch.tensor([[episode + step % 10 for _ in range(4)]])
    counter = torch.tensor([[episode]])
    return {"fric": fric, "stiff": stiff, "counter": counter}


class RealTimePlotter:
    def __init__(self, feet_labels, max_timesteps):
        # Create the application instance
        self.app = QtWidgets.QApplication([])

        # Create window
        self.win = pg.GraphicsLayoutWidget(show=True,title="Real-time plot of foot sensors")
        self.win.resize(1200, 1000)
        self.win.setBackground('w')  # Set background to white
        self.max_timesteps = max_timesteps  # Maximum number of timesteps on the x-axis
        self.feet_labels = feet_labels
        
        # Create plots and curves for each sensor
        self.plots = {}
        self.legends = {}
        self.curves = {}
        self.data = {}
        self.pred_curves = {}
        self.pred_data = {}
        self.error_text_items = {}
        self.total_timesteps={}
        self.terrain_bars = {}
        self.step=0
        self.episode=None
        self.display_mean=True
        # Define a list of colors
        self.colors = [
                        (70, 70, 120), 
                        (120, 70, 70), 
                        (70, 120, 70), 
                        (120, 120, 70),
                        (220, 20, 60),
                        (255, 165, 0),
                        (75, 0, 130),
                        (238, 130, 238),
                        (107, 142, 35),
                        (0, 255, 255),
                        (255, 215, 0),
                        (255, 105, 180),
                        (64, 224, 208),
                        (139, 69, 19)
                    ]

        for i, foot in enumerate(feet_labels):
            for j, attr in enumerate(["fric", "stiff"]):
                
                p = self.win.addPlot(title=f"{foot} {attr}")
               
                p.showGrid(x=True, y=True)
                p.setLabels(left="Value", bottom="Timestep")
                
                # Set y-axis range based on the attribute
                if attr == "fric":
                    p.setYRange(-0.1, 1.1)
                    ticks = [i/5.0 for i in range(-1, 6)]  # -0.1 to 1.1 with step of 0.1
                    p.getAxis('left').setTicks([[(val, f"{val:.2f}") for val in ticks]])
                elif attr == "stiff":
                    p.setYRange(0, 11)
                    ticks = generate_ticks(1, 10, 3)  # Assuming you want ticks every integer for stiffness
                    p.getAxis('left').setTicks([ticks])

                # curve = p.plot(pen=pg.mkPen(color=(i * 70, j * 70, 120)))
                self.plots[f"{foot}_{attr}"] = p
                self.curves[f"{foot}_{attr}"] = {}
                self.data[f"{foot}_{attr}"] = {}
                key = f"{foot}_{attr}_pred"
                self.pred_curves[key] = {}
                self.pred_data[key] = {}
                self.total_timesteps[f"{foot}_{attr}"]={}
                if j == 0:
                    self.win.nextColumn()
            self.win.nextRow()

        if self.display_mean:
            # self.win.nextRow()  # Move to the next row for the mean plots
            
            self.mean_labels = ['MEAN_FRIC', 'MEAN_STIFF']

            for label in self.mean_labels:
                # ... [code to create the mean plots] ...
                p = self.win.addPlot(title=label)
                p.showGrid(x=True, y=True)
                p.setLabels(left="Value", bottom="Timestep")
                
                if label == 'MEAN_FRIC':
                    p.setYRange(-0.1, 1.1)
                    ticks = [i/5.0 for i in range(-1, 6)]  # -0.1 to 1.1 with step of 0.1
                    p.getAxis('left').setTicks([[(val, f"{val:.2f}") for val in ticks]])
                    self.win.nextColumn() 
                elif label == 'MEAN_STIFF':
                    p.setYRange(0, 11)
                    ticks = generate_ticks(1, 10, 3)
                    p.getAxis('left').setTicks([ticks])

                self.plots[label] = p
                self.curves[label] = {}
                self.data[label] = {}
                self.pred_curves[label] = {}
                self.pred_data[label] = {}
               
         # Add a single unified legend:
        self.win.nextRow()
        legend_plot = self.win.addPlot()  # Create an empty plot for the legend
        self.global_legend = legend_plot.addLegend(offset=(10,10))  # Use rowCount=1 for horizontal orientation
        legend_plot.hideAxis('left')  # Hide axes and other plot elements
        legend_plot.hideAxis('bottom')
        legend_plot.setMenuEnabled(False)
        legend_plot.setMouseEnabled(x=False, y=False)
        
        self.legend_added = {}  # To keep track of which episode legends we've already added
        self.win.show()

        # # Set up a flag and QTimer to check the plotter window's visibility
        # self.plotter_ready = False
        # self.check_plotter_timer = QTimer()
        # self.check_plotter_timer.timeout.connect(self.check_plotter_status)
        # self.check_plotter_timer.start(100) 

    def check_plotter_status(self):
        # Check if the plotter window is visible or any other conditions
        if self.win.isVisible():
            self.plotter_ready = True
            self.check_plotter_timer.stop()  # Stop the timer once the window is visible


    def update_plot(self, single_env_obs):
        episode_number=0
        color = self.colors[episode_number % len(self.colors)]
        fric_mean_data = []
        stiff_mean_data = []
        for i, foot in enumerate(self.feet_labels):
            for j, attr in enumerate(["fric", "stiff"]):
                new_data_all_feet = single_env_obs[attr]
                if attr=="stiff":
                    new_data = new_data_all_feet[0, i]
                else:
                    new_data = new_data_all_feet[0, i]
                
                key = f"{foot}_{attr}"
                
                # Check if this episode's curve already exists. If not, create it.
                if episode_number not in self.curves[key]:
                    self.curves[key][episode_number] = self.plots[key].plot(pen=pg.mkPen(color=color, width=2))
                  
                # If the data for this episode doesn't exist yet, create it.
                if episode_number not in self.data[key]:
                    self.data[key][episode_number] = np.array([])
                
                if episode_number not in self.total_timesteps[key]:
                    self.total_timesteps[key][episode_number] = []

                self.total_timesteps[key][episode_number].append(self.step)
                self.data[key][episode_number] = np.append(self.data[key][episode_number], new_data)

                # Update the data of the current episode's curve
                self.curves[key][episode_number].setData(self.total_timesteps[key][episode_number], self.data[key][episode_number])
                # Add the episode to the global legend, if not already added:
                if episode_number not in self.legend_added:
                    self.global_legend.addItem(pg.PlotDataItem(pen=pg.mkPen(color=color, width=2)), f"Episode {episode_number}")
                    self.legend_added[episode_number] = True

                # self.add_terrain_bar(foot, attr, self.step, new_data)
            fric_mean_data.append(single_env_obs["fric"][0, i])
            stiff_mean_data.append(single_env_obs["stiff"][0, i])
        self.episode=episode_number
        # Calculate and plot the mean for friction and stiffness
        mean_fric = np.mean(fric_mean_data)
        mean_stiff = np.mean(stiff_mean_data)
        if self.display_mean:
            for label, mean_value in zip(self.mean_labels, [mean_fric, mean_stiff]):
                key = label
                if self.episode not in self.data[key]:
                    self.data[key][self.episode] = np.array([])
                    color = self.colors[self.episode % len(self.colors)]
                    self.curves[key][self.episode] = self.plots[key].plot(pen=pg.mkPen(color=color, width=2))
                self.data[key][self.episode] = np.append(self.data[key][self.episode], mean_value)
                self.curves[key][self.episode].setData(self.total_timesteps[f"{foot}_fric"][self.episode], self.data[key][self.episode])

        self.step += 1
        # Refresh the app
        # print("Qt processEvents thread ID:", threading.get_ident())
        QtWidgets.QApplication.processEvents()

    def update_predictions(self, predictions,avgs, id,plot_last=True): 
        # Lists to store mean predictions for friction and stiffness
        mean_fric_preds = []
        mean_stiff_preds = []
        fric_errors = []
        stiff_errors = []

        for i, foot in enumerate(self.feet_labels):
            for j, attr in enumerate(["fric", "stiff"]):
                pred_key = f"{foot}_{attr}_pred"
                data_key = f"{foot}_{attr}"
                pred_data_all_feet = predictions[attr]
                pred_data = pred_data_all_feet[id, i]
                episode = self.episode

                # If the episode's data for predictions doesn't exist yet, create it.
                if episode not in self.pred_data[pred_key]:
                    self.pred_data[pred_key][episode] = np.array([])
                    color = self.colors[episode % len(self.colors)+1]
                    self.pred_curves[pred_key][episode] = self.plots[data_key].plot(pen=pg.mkPen(color=color, width=2))

                self.pred_data[pred_key][episode] = np.append(self.pred_data[pred_key][episode], pred_data)
                self.pred_curves[pred_key][episode].setData(self.total_timesteps[data_key][episode], self.pred_data[pred_key][episode])

                # Store the predictions for calculating the mean
                if attr == "fric":
                    mean_fric_preds.append(pred_data)
                elif attr == "stiff":
                    mean_stiff_preds.append(pred_data)

                # Calculate the mean predictions for friction and stiffness
        mean_fric_pred = np.mean(mean_fric_preds)
        mean_stiff_pred = np.mean(mean_stiff_preds)

        # Update and plot the mean predictions
        for label, mean_pred in zip(self.mean_labels, [mean_fric_pred, mean_stiff_pred]):
            pred_key = label
            data_key = label  # Using the label directly since it matches the keys used for mean data

            if self.episode not in self.pred_data[pred_key]:
                self.pred_data[pred_key][self.episode] = np.array([])
                color = self.colors[self.episode % len(self.colors)]
                self.pred_curves[pred_key][self.episode] = self.plots[data_key].plot(pen=pg.mkPen(color=color, width=1, style=QtCore.Qt.DashLine))

            self.pred_data[pred_key][self.episode] = np.append(self.pred_data[pred_key][self.episode], mean_pred)
            self.pred_curves[pred_key][self.episode].setData(self.total_timesteps[f"{foot}_fric"][self.episode], self.pred_data[pred_key][self.episode])
        # update using average data
        if not plot_last:
            # Extract the length of predictions
            num_timesteps = avgs["fric"].shape[0]
            # Calculate the starting index for replacing data in the current episode
            start_replace_idx =  len(self.pred_data[next(iter(self.pred_data))][self.episode]) - num_timesteps
            offset=0
            # Check if we need to replace data in the previous episode
            if start_replace_idx < 0 and self.episode - 1 in self.pred_data[next(iter(self.pred_data))]:
                previous_episode_replace_idx = len(self.pred_data[next(iter(self.pred_data))][self.episode - 1]) + start_replace_idx
                # Replace in the previous episode
                for t in range(previous_episode_replace_idx, len(self.pred_data[next(iter(self.pred_data))][self.episode - 1])):
                    for i, foot in enumerate(self.feet_labels):
                        for attr in ["fric", "stiff"]:
                            pred_key = f"{foot}_{attr}_pred"
                            pred_data = avgs[attr][ t - previous_episode_replace_idx,id,i]
                            self.pred_data[pred_key][self.episode - 1][t] = pred_data

                # Adjust the starting index for the current episode
                start_replace_idx=0
                offset = len(self.pred_data[next(iter(self.pred_data))][self.episode - 1]) - previous_episode_replace_idx
            elif start_replace_idx<0 and self.episode==0:
                start_replace_idx=0
            # Replace the current episode data
            for t in range(start_replace_idx, len(self.pred_data[next(iter(self.pred_data))][self.episode])):
                for i, foot in enumerate(self.feet_labels):
                    for attr in ["fric", "stiff"]:
                        pred_key = f"{foot}_{attr}_pred"
                        pred_data = avgs[attr][t+offset-start_replace_idx ,id,i]
                        self.pred_data[pred_key][self.episode][t] = pred_data
        # Compute the mean prediction errors
        for i, foot in enumerate(self.feet_labels):
            for attr in ["fric", "stiff"]:
                pred_key = f"{foot}_{attr}_pred"
                data_key = f"{foot}_{attr}"
                
                all_pred_data = np.concatenate([self.pred_data[pred_key][ep] for ep in self.pred_data[pred_key]])
                all_true_data = np.concatenate([self.data[data_key][ep] for ep in self.data[data_key]])
                mean_error = np.mean(np.abs(all_pred_data - all_true_data))
                std_error = np.std(all_pred_data - all_true_data)
                # Create or update the TextItem
                if (foot, attr) in self.error_text_items:
                    # Update the text of the existing TextItem
                    self.error_text_items[(foot, attr)].setText(f"Mean Error: {mean_error:.3f}\nStd Error: {std_error:.3f}")
                else:
                    # Create a new TextItem and store its reference
                    text_item = pg.TextItem(text=f"Mean Error: {mean_error:.3f}\nStd Error: {std_error:.3f}", anchor=(0,1), color='red')
                    self.plots[data_key].addItem(text_item)
                    self.error_text_items[(foot, attr)] = text_item

        # Refresh the app
        QtWidgets.QApplication.processEvents()
    def add_terrain_bar(self, foot, attr, timestep, terrain_value):
        key = f"{foot}_{attr}_bar"
        
        if attr == "fric":
            y_position = -0.05
            height = 0.05
        else:  # attr == "stiff"
            y_position = 0
            height = 0.5  # Adjust this value if you want the bar to take up a different portion of the vertical space for stiffness
            terrain_value=terrain_value/ 10.0
        bar_item = pg.BarGraphItem(
            x=[timestep], 
            height=[height], 
            width=1, 
            brush=pg.mkBrush(self.get_terrain_color(terrain_value)),
            y0=[y_position],  # Setting the starting y position for the bar
            pen=None  # This will remove the outline around the bar
        )
        
        # Check if the key exists in the terrain_bars dictionary
        if key not in self.terrain_bars:
            self.terrain_bars[key] = []
        
        # Add the bar_item to the list and to the plot
        self.terrain_bars[key].append(bar_item)
        self.plots[f"{foot}_{attr}"].addItem(bar_item)

    def get_terrain_color(self, terrain_value):
        # Convert terrain value to a color using jet cmap. 
        # Here's a basic example. Depending on the range and nature of your terrain_value, you might need to adjust this:
        colormap = plt.get_cmap('jet')
        normed_value = terrain_value  # normalize if necessary
        rgba = colormap(normed_value)
        return rgba[0] * 255, rgba[1] * 255, rgba[2] * 255

    def run(self):
        # This method can be used to keep the GUI running
        if not QtWidgets.QApplication.instance():
            self.app = QtWidgets.QApplication([])
        QtWidgets.QApplication.instance().exec_()

    def close(self):
        # Close the plots
        self.win.close()

    def save_plot(self, filename="output.png"):
        """
        Save the current plot to an image file.

        Parameters:
        - filename: Name of the output file. Should include the desired file extension (e.g., '.png', '.jpg').
        """
        exporter = pg.exporters.ImageExporter(self.win.scene())
        exporter.export(filename)


# Example usage:
if __name__ == '__main__':
    plotter = RealTimePlotter(['FOOT_LF', 'FOOT_RF', 'FOOT_LH', 'FOOT_RH'], 30)
    
    # Simulate 3 episodes
    for episode in range(5):
        for step in range(30):
            obs = generate_mock_obs(step, episode)
            plotter.update_plot(obs)
    
    plotter.run()
