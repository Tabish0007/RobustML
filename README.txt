RobustML: Adversarial Robustness Benchmarking Toolkit
Overview

RobustML is a framework for evaluating machine learning models under adversarial attacks and defenses. The goal is to provide a standardized way to benchmark models across domains (vision, text, tabular) and help practitioners build more reliable AI systems.

Features

Plug-and-play support for PyTorch, TensorFlow, and scikit-learn models

Multiple adversarial attack implementations (FGSM, PGD, DeepFool, CW, TextAttack, etc.)

Defense methods (adversarial training, denoising autoencoders, randomized smoothing)

Benchmarking pipeline with standardized metrics

Visualization of adversarial samples and defense performance

Extendable design for new attacks, defenses, and datasets

Installation
git clone https://github.com/yourusername/robustml.git
cd robustml
pip install -r requirements.txt

Usage
1. Run a quick benchmark
python benchmark.py --model my_model.pth --dataset cifar10

2. Apply an adversarial attack
python attack.py --attack fgsm --model my_model.pth --dataset mnist

3. Apply a defense
python defense.py --defense adversarial_training --model my_model.pth --dataset cifar10

4. Visualization example
python visualize.py --attack pgd --dataset mnist

Supported Components
Attacks

FGSM (Fast Gradient Sign Method)

PGD (Projected Gradient Descent)

DeepFool

Carlini-Wagner

TextAttack (for NLP)

Defenses

Adversarial Training

Input Preprocessing (denoising, JPEG compression)

Denoising Autoencoders

Randomized Smoothing

Datasets

Vision: MNIST, CIFAR-10, Tiny ImageNet

Text: IMDB Sentiment, Toxic Comment

Tabular: Credit Card Fraud Detection

Benchmark Metrics

Clean Accuracy

Robust Accuracy (accuracy under attack)

Attack Success Rate

Defense Efficiency

Computational Cost

Project Structure
robustml/
│── attacks/
│   ├── fgsm.py
│   ├── pgd.py
│   ├── deepfool.py
│── defenses/
│   ├── adv_training.py
│   ├── smoothing.py
│── datasets/
│   ├── mnist.py
│   ├── cifar10.py
│── utils/
│   ├── visualization.py
│   ├── metrics.py
│── benchmark.py
│── attack.py
│── defense.py
│── visualize.py
│── requirements.txt
│── README.md

Roadmap

 Add support for ImageNet scale datasets

 Expand to multimodal (image+text) adversarial robustness

 Include certified robustness evaluation

 Build a Streamlit web app demo  

Citation

If you use RobustML with my permission  in your research or project, please cite:

@misc{robustml2025,
  author = {Your Name},
  title = {RobustML: Adversarial Robustness Benchmarking Toolkit},
  year = {2025},
  url = {https://github.com/yourusername/robustml}
}
