import gradio as gr
import cv2
import numpy as np
import ollama
from skimage.morphology import skeletonize
import sys

# --- CORE LOGIC FUNCTIONS ---

def calculate_crae_crve(image):
    if image is None:
        return 0, 0, None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
    height, width = image.shape[:2]
    target_width = 800
    scale = target_width / width
    resized_img = cv2.resize(image, (target_width, int(height * scale)))
    
    b, g, r = cv2.split(resized_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(g)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(enhanced_g, cv2.MORPH_OPEN, kernel)
    vessel_mask = cv2.adaptiveThreshold(
        opening, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # REFINED SKELETONIZATION
    skeleton_bool = skeletonize(vessel_mask.astype(bool))
    skeleton = (skeleton_bool * 255).astype(np.uint8)

    dist_transform = cv2.distanceTransform(vessel_mask, cv2.DIST_L2, 5)
    vessel_widths = dist_transform[skeleton == 255] * 2
    vessel_widths = vessel_widths[vessel_widths > 1.5]

    if len(vessel_widths) == 0:
        return 0, 0, skeleton

    sorted_widths = np.sort(vessel_widths)
    split_idx_low = int(len(sorted_widths) * 0.3)
    split_idx_high = int(len(sorted_widths) * 0.7)
    
    raw_crae_px = np.mean(sorted_widths[:split_idx_low])
    raw_crve_px = np.mean(sorted_widths[split_idx_high:])

    calibration_factor = 45.0 
    crae = round(raw_crae_px * calibration_factor, 2)
    crve = round(raw_crve_px * calibration_factor, 2)

    return crae, crve, skeleton

def medical_diagnosis(crae, crve):
    prompt = f"""
    You are a Retinal Diagnostic AI. Analyze:
    CRAE: {crae} µm (Normal: 196 ± 13)
    CRVE: {crve} µm (Normal: 220 ± 15)
    
    Provide: 1. Biometric Audit, 2. Risk Assessment, 3. Potential Indications.
    Keep it strictly clinical. Consult a professional.
    """
    try:
        response = ollama.chat(model='medllama2:latest', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Ollama Connection Error: Ensure Ollama app is running and 'medllama2' is pulled."

def process_pipeline(image):
    crae, crve, skeleton = calculate_crae_crve(image)
    report = medical_diagnosis(crae, crve)
    metrics_text = f"### 📊 Vessel Measurements\n| Metric | Value |\n| :--- | :--- |\n| **CRAE** | {crae} µm |\n| **CRVE** | {crve} µm |"
    return metrics_text, report, skeleton

# --- GRADIO INTERFACE BLOCK ---

with gr.Blocks() as demo:
    gr.Markdown("# 👁️ Retinal Vessel Analysis")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Scan", type="numpy")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            mask_output = gr.Image(label="Vessel Map")
            metrics_output = gr.Markdown()
            ai_output = gr.Markdown()

    submit_btn.click(fn=process_pipeline, inputs=img_input, outputs=[metrics_output, ai_output, mask_output])

# --- CRITICAL SCRIPT EXECUTION SECTION ---

if __name__ == "__main__":
    try:
        print("Initializing local server...")
        # 'show_error=True' helps debug if the script fails inside the browser
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        sys.exit(0)