# The Ultimate Beginner's Guide to Remote GPU Grid Searching

# The Ultimate Guide to Remote GPU Grid Searching

> **The Easy Way:** We have created a fully automated script that handles this entire process for you. For the simplest experience, please use it!
> 
> ```bash
> # Run this command from the project root
> python remote/run_remote_experiment.py
> ```
> 
> This script will ask for your server details and config file, then run all the steps below automatically. The rest of this guide serves as a detailed, manual walkthrough for those who want to understand each step of the process.

---

Welcome! This guide will walk you through every step of running a powerful, automated hyperparameter search on a remote GPU server. Even if you've never used a remote server, `ssh`, or `git` before, this tutorial will make the process simple and clear.

**What you will accomplish:** You will rent a powerful GPU in the cloud, automatically set it up with a single command, run a complex experiment that would be too slow for your laptop, and download the results for analysis.

---

## Part 1: Setting Up Your Local Machine (One-Time Setup)

Before we can talk to the remote server, we need to set up a few tools on your own computer.

### 1.1: Install Git
*   **What it is:** Git is a tool for managing and downloading code.
*   **How to install:**
    *   **Windows/macOS:** [Download Git here](https://git-scm.com/downloads).
    *   **Linux:** Open a terminal and run `sudo apt-get install git`.
*   **Verify installation:** Open a terminal and run `git --version`. You should see a version number.

### 1.2: Install Conda
*   **What it is:** Conda is a tool for managing Python versions and libraries.
*   **How to install:** Download and install the **Miniconda** installer from [this page](https://docs.conda.io/en/latest/miniconda.html). It's small and has everything we need.
*   **Verify installation:** Open a new terminal and run `conda --version`.

### 1.3: Get the Project Code
*   Now, let's download the code for this project from GitHub.
*   Open a terminal, navigate to where you want to store the project, and run:
    ```bash
    git clone https://github.com/WillForEternity/MemoryCapacitorTopologies.git
    ```
*   Move into the newly created project directory:
    ```bash
    cd MemoryCapacitorTopologies
    ```

### 1.4: Install Local Helper Tools
*   Our automation scripts need a couple of small Python libraries to work.
*   In your terminal (inside the project folder), run:
    ```bash
    python -m pip install pyyaml paramiko
    ```

---

## Part 2: Setting Up Your Remote GPU Server

Now we'll rent a GPU server from a service called RunPod.

### 2.1: Create a RunPod Account
*   Go to [runpod.io](https://runpod.io) and create an account. You will need to add some credits to rent a server.

### 2.2: Generate an SSH Key
*   **What it is:** An SSH key is like a very secure password that allows your computer to connect to the remote server without you typing a password each time.
*   Open a terminal on your local machine and run the following command. Replace the email with your own.
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```
*   Press **Enter** three times to accept the default location and skip setting a passphrase.
*   Now, view and copy your new key by running:
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
    Select and copy the entire output, which starts with `ssh-ed25519` and ends with your email.

### 2.3: Add Your SSH Key to RunPod
*   In RunPod, go to **Settings -> SSH Keys**.
*   Click **New Key**, give it a name (e.g., "My Laptop"), paste the key you just copied, and save it.

### 2.4: Deploy a GPU Pod
*   Go to **Secure Cloud** and click **Deploy**.
*   Select a powerful GPU like the **RTX A6000**.
*   In the "Template" search bar, find and select **RunPod PyTorch 2**.
*   Under "Customize Deployment," select your newly added SSH key.
*   Click **Deploy**. After a minute, your pod will appear in the **My Pods** section with a green "Running" status.

### 2.5: Get Your Connection Details
*   On the **My Pods** page, find your new pod and click the **Connect** button. A window will pop up showing you how to connect via SSH. It will look something like this:
    `ssh root@157.157.221.29 -p 23202`
*   Keep these details handy. You'll need the IP address (`157.157.221.29`) and the port (`23202`).

---

## Part 3: The Automated Workflow

Now we'll connect everything and run the experiment.

### 3.1: Configure the Connection File (`pods.yml`)
*   This file tells our scripts how to find your remote server.
*   Open `remote/pods.yml` in a text editor. Update the `uri`, `port`, and `key` to match your pod's details from step 2.5.
    ```yaml
    # ... (other settings)
    pods:
      gpu1:
        uri: "root@157.157.221.29"  # <-- Your pod's IP address
        port: 23202                  # <-- Your pod's port number
        key: "~/.ssh/id_ed25519"      # <-- Should already be correct
        role: gpu
    ```

### 3.2: Provision the Server (The Magic Step)
*   This single command connects to your server and automatically sets up everything: Conda, Python, all dependencies, and the latest code from GitHub.
*   On your **local machine**, run:
    ```bash
    python remote/setup.py
    ```
*   This will take several minutes. Be patient! It's finished when you see the message: `[✔] gpu1 ready`.

### 3.3: Launch the Grid Search
*   Now for the main event! This command starts the experiment on the remote server, using the robust settings we discovered to ensure true parallelism.
*   On your **local machine**, run the `ssh` command from step 2.5, but add the experiment command at the end. **Make sure to replace the user, IP, and port with your own.**
    ```bash
    ssh -t -i ~/.ssh/id_ed25519 -p 23202 root@157.157.221.29 "cd /root/MemoryCapacitorTopologies && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /root/miniconda/envs/rc/bin/python -u experiments/grid_search.py configs/lorenz_search.yaml"
    ```
*   You will see a real-time stream of `[✔] FINISHED` messages as the jobs complete. This will run for a while.

> **⚠️ Important Note on the SSH Command:**
> The `ssh` command in step 3.3 is very long. When you copy it, make sure you replace **only** the user, IP address, and port number with your own details. The rest of the command, especially the part in quotes, must be copied exactly as it is.

### 3.4: Retrieve Your Results
*   Once the grid search is done, all results are on the remote server. Let's download them.
*   On your **local machine**, run:
    ```bash
    python remote/pull_outputs.py
    ```
*   This copies the `training/outputs/` directory from the remote server to your local project folder.

### 3.5: Analyze the Interactive Plot
*   The results are now on your computer!
*   Navigate to `training/outputs/lorenz_search_results/`.
*   Find and double-click `lorenz_predictions_3d.html` to open it in your browser.
*   You can now explore the 3D plot of the Lorenz attractor your model predicted!

---

## Part 4: Troubleshooting & FAQ

*   **Error: `Permission denied (publickey)`**
    *   **Cause:** Your SSH key isn't set up correctly in RunPod, or `pods.yml` is pointing to the wrong key file.
    *   **Solution:** Go back to steps 2.2 and 2.3. Ensure you've copied the *entire* public key into RunPod. Verify the `key` path in `pods.yml` is `~/.ssh/id_ed25519`.

*   **Error: `Connection timed out` when running `setup.py`**
    *   **Cause:** Your pod is not running, or the `uri`/`port` in `pods.yml` is incorrect.
    *   **Solution:** Go to the **My Pods** page on RunPod and make sure your pod has a green "Running" status. Double-check that the IP and port in `pods.yml` match the connection details exactly.

*   **The grid search runs very slowly or seems to hang.**
    *   **Cause:** You forgot to include the `OMP_NUM_THREADS=1 ...` part of the command in step 3.3.
    *   **Solution:** This part is critical! It prevents your CPU cores from fighting each other. Copy the command from step 3.3 exactly as written.

*   **Error: `No such file or directory` when running `pull_outputs.py`**
    *   **Cause:** The grid search either failed or didn't finish.
    *   **Solution:** Re-run the command from step 3.3 and wait for it to print "--- Grid Search Complete ---" before trying to pull the results.

---

## Part 5: Shutting Down Your Server (To Save Money!)

Once you have successfully downloaded your results, it is **very important** to shut down your remote server so you don't continue to be charged for it.

1.  Go to the **My Pods** page on RunPod.
2.  Find your pod, click the three-dots menu icon.
3.  Select **Terminate Pod**.

This will permanently delete the server and stop all charges.
