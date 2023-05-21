import numpy as np 
import matplotlib.pyplot as plt 
import os 
import paramiko





x = np.arange(1,2,0.1)
y = np.sin(np.pi * x)

plt.plot(x,y)
plt.show()
plt.savefig("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/fig.png")
ssh = paramiko.SSHClient()
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.connect("rose", username="migo", password= "palaver120,")
sftp = ssh.open_sftp()
sftp.put("/home/migo/TUHP/Automatic-Epileptic-Seizure-Detection/fig.png", "/Users/toucanfirm/Documents/DTU/Speciale/SSH figures")
sftp.close()
ssh.close()