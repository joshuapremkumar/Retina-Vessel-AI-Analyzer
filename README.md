# 👁️ Retinal Vessel Analysis System

## 🚀 Overview
A diagnostic support tool developed for retinal vessel analysis. It automates the calculation of **CRAE** and **CRVE** metrics from fundus images to predict risks of Hypertension, Stroke, and Ischemia.

## 🛠️ Features
- **Computer Vision**: Zhang-Suen Skeletonization for precise vessel mapping.
- **Medical AI**: Integration with `MedLLaMA2` via Ollama for clinical interpretation.
- **UI**: Interactive Gradio dashboard.
- 
- Precision Segmentation: Uses Adaptive Gaussian Thresholding and CLAHE (Contrast Limited Adaptive Histogram Equalization) to isolate vessels even in low-contrast fundus images.

Zhang-Suen Skeletonization: A robust thinning algorithm that preserves the topological connectivity of the vascular network.

Local AI Interpretation: Integrated with MedLLaMA2 via Ollama, ensuring patient data privacy by processing all clinical interpretations locally (No Cloud API required).

Automated Biometrics: Calculates:

    CRAE (Central Retinal Arteriolar Equivalent): Narrowing indicates potential hypertension.

    CRVE (Central Retinal Venular Equivalent): Dilation indicates potential inflammation or stroke risk.

## 📦 Installation
1. Install Ollama and pull MedLLaMA2: `ollama pull medllama2`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python retina_analysis.py`

📑 Clinical Logic & Python Implementation
The script follows a 3-stage pipeline to ensure data integrity:
1. Image Pre-processing & Enhancement
The code extracts the Green Channel of the RGB image because it provides the highest contrast between the blood vessels and the retinal background. It then applies CLAHE to normalize lighting across the image.
2. Vessel Width Calculation
Instead of simple pixel counting, the system uses a Euclidean Distance Transform.
•	The code identifies the "Skeleton" (1-pixel wide centerline).
•	It measures the distance from each skeleton pixel to the nearest edge of the vessel.
•	Formula: $Width = Distance \times 2 \times CalibrationFactor$.
3. Artery/Vein Heuristic
Since arteries and veins are distinct in thickness, the code uses a statistical split:
•	Bottom 30% of measured widths are classified as Arteries.
•	Top 30% of measured widths are classified as Veins.
•	This removes mid-range noise and focuses on the most significant vessels for CRAE/CRVE calculation.
________________________________________
📦 Installation & Setup
1. Model Setup
Install Ollama and download the medical-grade LLM:
Bash
ollama pull medllama2
2. Environment Setup
Clone the repository and install the required Python libraries:
Bash
git clone https://github.com/joshuapremkumar/Retina-Vessel-AI-Analyzer.git
cd Retina-Vessel-AI-Analyzer
pip install -r requirements.txt
3. Execution
Run the Gradio interface:
Bash
python retina_analysis.py
________________________________________
⚖️ Medical Disclaimer
IMPORTANT: This tool is provided for educational and research purposes as part of a hackathon submission. It is not a cleared medical device and is not intended for clinical use or to replace the judgment of a qualified ophthalmologist or cardiologist. Always consult a medical professional for diagnosis.


