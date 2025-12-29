# 👁️ Retinal Vessel Analysis System

## 🚀 Overview
A diagnostic support tool developed for retinal vessel analysis. It automates the calculation of **CRAE** and **CRVE** metrics from fundus images to predict risks of Hypertension, Stroke, and Ischemia.

## 🛠️ Features
- **Computer Vision**: Zhang-Suen Skeletonization for precise vessel mapping.
- **Medical AI**: Integration with `MedLLaMA2` via Ollama for clinical interpretation.
- **UI**: Interactive Gradio dashboard.

## 📦 Installation
1. Install Ollama and pull MedLLaMA2: `ollama pull medllama2`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python retina_analysis.py`

