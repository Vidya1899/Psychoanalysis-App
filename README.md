# Psychoanalysis App
## A non-intrusive mobile application to analyze human handwriting samples
![](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/Cloud-GCP-informational?style=flat&logo=cpp&logoColor=white&color=2bbc8a)
![](https://img.shields.io/badge/OS-Linux-informational?style=flat&logo=linux&logoColor=white&color=2bbc8a)
<!-- TABLE OF CONTENTS -->
### Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Web Hosting](#web-hosting)
* [Citation](#citation)
* [Contributing](#contributing)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project
The app is an automated graphology system, that will analyze human handwriting samples using computer vision and natural language processing techniques. Through the visual feature analysis using computer vision, the proposed system will generate a personality trait map using a coarse/high-levelclassification of the basic behavioral and psychological characteristics of the user. A high resolution visual feature analysis will beconducted on the handwriting samples to report on the quality of the strokes, the grammar correctness of the sentences, and the contextual referencing of the sentences and paragraphs. These features and traits generated will be compared with potential markers identified across different mental health conditions E.g. letter mirroring in dyslexia, change instroke sizes in personality analysis and going back-forth under depression. Overall, the key outcome of the app will be a detailed report summarizing the user’s psychological traits, which could be useful as an early indicator that the user needs medical intervention.

|         Resource    |       LINK       |
|:-------------------:|:----------------:|
|Web-App              |    [App](http://34.122.250.178/)     |
|Dataset           |    Personality Analysis, [Dyslexia](https://drive.google.com/drive/folders/1S895_SOM9YqAUS1mp1sKV-_JzzeXsau0?usp=sharing) , [Depression](https://drive.google.com/drive/folders/1MBF8T-XxV5X1jw8oAAElGBU-63hV-4oy?usp=sharing)|
|Others          |    Paper, [Data Collection](https://docs.google.com/document/d/1OT4qCCaR5KvNP0_dipuaWHn8UVHSaNOt6a9WKLfH7mU/edit?usp=sharing), BlogPost      |

## Web Hosting
### Initial Setup
1. Create a free-tier VM instance from the GCP compute engine console
2. Configure gcloud setup by following [these instructions](https://cloud.google.com/deployment-manager/docs/step-by-step-guide/installation-and-setup)
3. SSH into your pod by using the gcloud command available in the console in the VM instances page.

![alt-text-1](forReadme/sshCommand.png "ssh")

4. Make sure the port 80 is open for HTTP, TCP protocol.
   <p align="center">
    <img src="forReadme/network.png" alt="alt text"width="600" height="200">
    </p>
5. Install flask, and other dependencies required to run our inferencing scripts
6. Clone [Psychoanalysis-App](https://github.com/Vidya1899/Psychoanalysis-App) repository
7. To start the server. Change directory to our repository and run `sudo python3 app.py &`

### Changes Deployment  
1. Pushing the commits from local system to the main branch.  
2. SSH into your pod by using the gcloud command available in the console.
3. cd to our repo and do a `git pull`
4. Now kill the current server process and restart so that the new changes reflect in the production.
   ```
   ps -ax | grep app.py
   #now note down the pid of our process and enter the following
   #command with our pid in it
   sudo kill -9 <pid>
   ```
    <p align="center">
    <img src="forReadme/process.png" alt="alt text"width="700" height="100">
    </p>
5. Run, `sudo python3 app.py &`

<!-- CITATION -->
## Citation
1. A comperative evaluation of MARL algorithms that includes this environment
```
@article{papoudakis2020comparative,
  title={Comparative Evaluation of Multi-Agent Deep Reinforcement Learning Algorithms},
  author={Papoudakis, Georgios and Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2006.07869},
  year={2020}
}
```
2. A method that achieves state-of-the-art performance in many Level-Based Foraging tasks
```
@inproceedings{christianos2020shared,
  title={Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
  author={Christianos, Filippos and Sch{\"a}fer, Lukas and Albrecht, Stefano V},
  booktitle = {Advances in Neural Information Processing Systems},
  year={2020}
}
```


<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
## Contact

MORSE STUDIO- mail@mail.com

Project Link: [https://github.com/Vidya1899/Psychoanalysis-App](https://github.com/Vidya1899/Psychoanalysis-App)


