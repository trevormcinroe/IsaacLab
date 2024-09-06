import matplotlib.pyplot as plt
import numpy as np
import os

from tbparse import SummaryReader


plt.rcParams.update(plt.rcParamsDefault)


plt.figure(figsize=(10, 6))
# Set global font properties
plt.rcParams['font.family'] = 'serif'  # Use serif fonts
# plt.rcParams['font.serif'] = ['Times New Roman']  # Specific serif font
plt.rcParams['font.size'] = 14  # Font size
plt.rcParams['text.usetex'] = True

plt.rcParams.update(plt.rcParamsDefault)

# Set DPI
dpi = 300

plot_raw = False

log_dir = "/workspace/isaaclab/IsaacLabExtension/logs/skrl"
output_dir = "/workspace/isaaclab/IsaacLabExtension/images"
# experiment_names = ["prop_cartpole_3seed", "cam_cartpole_5seed", "ConcatCartpole_3seed"]

# log_dir = "/home/elle/code/external/IsaacLab/IsaacLabExtension/logs/rl_games"
# output_dir = "/home/elle/code/external/IsaacLab/IsaacLabExtension/images"

cartpole_hybrid = {
    "title": "Cartpole with cart image and pole proprioception",
    "legend_names": ["Normal proprioception", "Normal image", "Cart image, pole proprioception", "Pole proprioception"], # , "Image", "Concat(P, Img)"],
    "experiment_names": ["skrl_cartpole_prop2", "skrl_cartpole_camera2", "concat_imgcart_proppole", "skrl_cartpole_missing_pole"],
    "output_file": "cartpole_hybrid.png",
}

cartpole_missing = {
    "title": "Cartpole with partial proprioception",
    "legend_names": ["Full", r'$x$ $\theta$', r'$\dot{x}$ $\dot{\theta}$', r'$x$ $\dot{x}$', r'$\theta$ $\dot{\theta}$'], # , "Image", "Concat(P, Img)"],
    "experiment_names": ["skrl_cartpole_prop2", "skrl_cartpole_missing_jointpos", "skrl_cartpole_missing_vel", "skrl_cartpole_missing_cart", "skrl_cartpole_missing_pole"],
    "output_file": "cartpole_missing.png",
}
cartpole_baseline = {
    "title": "Cartpole",
    "legend_names": ["Proprioception", "Image", "Concat(P, Img)"],
    "experiment_names": ["skrl_cartpole_prop2", "skrl_cartpole_camera2", "skrl_cartpole_concat2"],# "ConcatCartpole_3seed"],
    "output_file": "baseline.png",
}

cam_linear_discs = {
    "title": "Cartpole",
    "legend_names": ["Normal", "N=5", "N=10", "N=15"],
    "experiment_names": ["cam_cartpole_5seed", "cam-5-linear-discs", "cam-10-linear-discs", "cam-15-linear-discs"],
    "output_file": "/home/elle/Pictures/baselines_linear_discs.png",
}

cam_cartpole = {
    "title": "Camera Cartpole",
    # "legend_names": ["Image"],
    "experiment_names": ["cam_cartpole_5seed"],
    "output_file": "/home/elle/Pictures/cartpole_badseeds.png",
}

behaviour = "random"
cam_cartpole_occlusion = {
    "title": f"Camera Cartpole with {behaviour} discs",
    "legend_names": ["Normal", "N=5", "N=10", "N=15"],
    "experiment_names": ["CameraCartpole_final", f"CameraCartpole-{behaviour}-5", f"CameraCartpole-{behaviour}-10", f"CameraCartpole-{behaviour}-15"],
    "output_file": f"camera_{behaviour}_discs.png",
}


concat_occlusion = {
    "title": "Cartpole",
    "legend_names": ["Normal", "N=5", "N=10", "N=15"],
    "experiment_names": ["ConcatCartpole_3seed", f"concat-5-{behaviour}-discs", f"concat-10-{behaviour}-discs", f"concat-15-{behaviour}-discs"],
    "output_file": "/home/elle/Pictures/baselines_linear_discs.png",
}

exp_dict = cartpole_hybrid


for i, experiment_name in enumerate(exp_dict["experiment_names"]):

    experiment_dir = os.path.join(log_dir, experiment_name)
    runs = os.listdir(experiment_dir)
    if runs == []:
        raise ValueError("No runs in directory:", experiment_dir)
    # print(runs)

    # get min length
    min_length = 10000
    for run in runs:
        run_dir = os.path.join(experiment_dir, run)
        print(run_dir)
        event_file = sorted(os.listdir(run_dir))[1]
        event_file = os.path.join(run_dir, event_file)
        reader = SummaryReader(event_file)  # long format
        df = reader.scalars
        rewards = df[df["tag"] == "Reward / Total reward (mean)"]["value"].to_numpy()
        min_length = np.min((min_length, np.shape(rewards)[0]))
    print("min length", min_length)
    # min_length=100
    mean_rewards = []
    step = []
    for run in runs:
        run_dir = os.path.join(experiment_dir, run)
        event_file = sorted(os.listdir(run_dir))[1]
        event_file = os.path.join(run_dir, event_file)
        reader = SummaryReader(event_file)  # long format
        df = reader.scalars
        rewards = df[df["tag"] == "Reward / Total reward (mean)"]
        step = rewards["step"].to_numpy()[:min_length]
        run_rewards = rewards["value"].to_numpy()[:min_length]
        mean_rewards.append(run_rewards)
        print("Num rewards", np.shape(run_rewards))
        if plot_raw:
            plt.plot(step, run_rewards, label=exp_dict["legend_names"][i])

    # Plot the mean and standard deviation
    if not plot_raw:
        # Calculate the mean and standard deviation across the seeds
        mean = np.mean(mean_rewards, axis=0)
        std_dev = np.std(mean_rewards, axis=0)
        plt.plot(step, mean, label=exp_dict["legend_names"][i])
        print(exp_dict["legend_names"][i])
        plt.fill_between(step, mean - std_dev, mean + std_dev, alpha=0.2)

plt.ylim([0,350])
# plt.xlim([0,5e6])
plt.xlabel("Timestep")
plt.ylabel("Mean evaluation rewards")
plt.legend()
plt.title(exp_dict["title"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, exp_dict["output_file"]), dpi=dpi)
plt.show()
