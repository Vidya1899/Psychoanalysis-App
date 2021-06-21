# Psychoanalysis App
![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Cloud-GCP-informational?style=flat&logo=cpp&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=linux&logoColor=white&color=2bbc8a)

|         Resource    |       LINK       |
|:-------------------:|:----------------:|
|Web-App              |    [App](http://34.122.250.178/)     |
|Dataset           |    Personality Analysis, [Dyslexia](https://drive.google.com/drive/folders/1S895_SOM9YqAUS1mp1sKV-_JzzeXsau0?usp=sharing) , [Depression](https://drive.google.com/drive/folders/1MBF8T-XxV5X1jw8oAAElGBU-63hV-4oy?usp=sharing)|
|Others          |    Paper, [Data Collection](https://docs.google.com/document/d/1OT4qCCaR5KvNP0_dipuaWHn8UVHSaNOt6a9WKLfH7mU/edit?usp=sharing), BlogPost      |

### Initial Setup
1. Create a free-tier VM instance from the GCP compute engine console
2. Configure gcloud setup by following [these instructions](https://cloud.google.com/deployment-manager/docs/step-by-step-guide/installation-and-setup)
3. SSH into our pod by using the gcloud command available in the console in the VM instances page.

![alt-text-1](forReadme/sshCommand.png "ssh")

4. Make sure the port 80 is open for HTTP, TCP protocol.
   <p align="center">
    <img src="forReadme/network.png" alt="alt text"width="600" height="200">
    </p>
5. Install flask, and other dependencies required to run our inferencing scripts
6. Clone Graphalogy repository
7. To start the server. Change directory to our repository and run `sudo python3 app.py &`

### Changes Deployment  
1. Pushing the commits from local system to our main branch.  
2. SSH into our pod by using the gcloud command available in the console.
3. Now cd to our repo and do a `git pull`
4. Now we will have to kill the current server process and restart so that the new changes reflect in the production.
   ```
   ps -ax | grep app.py
   #now note down the pid of our process and enter the following
   #command with our pid in it
   sudo kill -9 <pid>
   ```
    <p align="center">
    <img src="forReadme/process.png" alt="alt text"width="700" height="100">
    </p>
5. Now run, `sudo python3 app.py &`
