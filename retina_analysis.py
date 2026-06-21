import gradio as gr
import cv2
import numpy as np
import ollama
from skimage.morphology import skeletonize
import sys
import json
import time
import urllib.request
import threading

PENDO_TRACK_URL = "https://data.pendo.io/data/track"
PENDO_INTEGRATION_KEY = "2b736422-7b4e-4721-9b6e-5259e83af672"


def _pendo_track(event_name, properties=None):
    """Send a track event to the Pendo server-side API in a background thread."""
    def _send():
        try:
            payload = {
                "type": "track",
                "event": event_name,
                "visitorId": "system",
                "accountId": "system",
                "timestamp": int(time.time() * 1000),
                "properties": properties or {}
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                PENDO_TRACK_URL,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "x-pendo-integration-key": PENDO_INTEGRATION_KEY
                },
                method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

# --- CORE LOGIC FUNCTIONS ---

def calculate_crae_crve(image):
    if image is None:
        _pendo_track("vessel_analysis_failed", {
            "failure_reason": "image_was_none",
            "image_was_none": True
        })
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
        _pendo_track("vessel_analysis_failed", {
            "failure_reason": "no_vessel_widths",
            "image_was_none": False,
            "resized_image_width": target_width,
            "resized_image_height": int(height * scale)
        })
        return 0, 0, skeleton

    _pendo_track("vessel_analysis_completed", {
        "vessel_widths_count": int(len(vessel_widths)),
        "resized_image_width": target_width,
        "resized_image_height": int(height * scale),
        "scale_factor": round(float(scale), 4),
        "min_vessel_width": round(float(np.min(vessel_widths)), 2),
        "max_vessel_width": round(float(np.max(vessel_widths)), 2),
        "mean_vessel_width": round(float(np.mean(vessel_widths)), 2)
    })

    sorted_widths = np.sort(vessel_widths)
    split_idx_low = int(len(sorted_widths) * 0.3)
    split_idx_high = int(len(sorted_widths) * 0.7)
    
    raw_crae_px = np.mean(sorted_widths[:split_idx_low])
    raw_crve_px = np.mean(sorted_widths[split_idx_high:])

    calibration_factor = 45.0 
    crae = round(raw_crae_px * calibration_factor, 2)
    crve = round(raw_crve_px * calibration_factor, 2)

    _pendo_track("biometric_metrics_calculated", {
        "crae_value": crae,
        "crve_value": crve,
        "calibration_factor": calibration_factor,
        "arteriolar_widths_count": split_idx_low,
        "venular_widths_count": int(len(sorted_widths) - split_idx_high),
        "total_vessel_widths_count": int(len(sorted_widths)),
        "crae_within_normal_range": 183 <= crae <= 209,
        "crve_within_normal_range": 205 <= crve <= 235
    })

    return crae, crve, skeleton

def medical_diagnosis(crae, crve):
    prompt = f"""
    SYSTEM ROLE:
You are a Retinal Diagnostic AI.
You DO NOT ask questions.
You DO NOT request clarification.
You ONLY generate the report using the rules below.

ASSUMPTIONS:
- CRAE and CRVE values are already computed correctly.
- Reference ranges provided below are final and authoritative.

REFERENCE NORMAL VALUES:
- Normal CRAE: 196 ± 13 µm (Normal range: 183–209 µm)
- Normal CRVE: 220 ± 15 µm (Normal range: 205–235 µm)

PATIENT MEASUREMENTS:
- CRAE: {crae} µm
- CRVE: {crve} µm

DECISION RULES (STRICT):
1. If CRAE < 183 µm → Arterial narrowing → Hypertension risk
2. If CRVE > 235 µm → Venular dilation → Inflammation / Diabetes / Ischemia risk
3. If both CRAE and CRVE are within normal range → No significant retinal vascular risk
4. Any deviation outside normal range → Declare AT RISK

TASK:
Generate a **concise clinical report** with EXACTLY these three sections
and NO additional sections:

1. **Biometric Audit**
   - State whether CRAE and CRVE are within or outside normal limits.

2. **Risk Assessment**
   - Output ONLY one of the following phrases:
     - "AT RISK"
     - "NO SIGNIFICANT RISK"

3. **Potential Indications**
   - Mention Hypertension if CRAE is low
   - Mention Stroke or Ischemia if CRVE is high
   - If no abnormalities, state "No abnormal vascular indications detected."

MANDATORY STATEMENT:
End the report with:
"Clinical correlation and consultation with a qualified medical professional is required."

STYLE RULES:
- Formal clinical tone
- No conversational language
- No questions
- No explanations of methodology
- No disclaimers beyond the mandatory statement

    """
    try:
        response = ollama.chat(model='medllama2:latest', messages=[{'role': 'user', 'content': prompt}])
        report_content = response['message']['content']
        _pendo_track("ai_clinical_report_generated", {
            "crae_input": crae,
            "crve_input": crve,
            "model_name": "medllama2:latest",
            "report_length_chars": len(report_content),
            "risk_status": "AT RISK" if "AT RISK" in report_content.upper() else "NO SIGNIFICANT RISK",
            "crae_below_normal": crae < 183,
            "crve_above_normal": crve > 235
        })
        return report_content
    except Exception as e:
        _pendo_track("ai_report_generation_failed", {
            "crae_input": crae,
            "crve_input": crve,
            "error_message": str(e)[:200],
            "model_name": "medllama2:latest"
        })
        return f"Ollama Connection Error: Ensure Ollama app is running and 'medllama2' is pulled."

def process_pipeline(image):
    _pendo_track("fundus_image_uploaded", {
        "image_height": int(image.shape[0]) if image is not None else 0,
        "image_width": int(image.shape[1]) if image is not None else 0,
        "image_channels": int(image.shape[2]) if image is not None and len(image.shape) > 2 else 1,
        "has_alpha_channel": bool(image is not None and len(image.shape) > 2 and image.shape[2] == 4),
        "is_grayscale": bool(image is not None and len(image.shape) == 2)
    })

    crae, crve, skeleton = calculate_crae_crve(image)
    report = medical_diagnosis(crae, crve)
    metrics_text = f"### 📊 Vessel Measurements\n| Metric | Value |\n| :--- | :--- |\n| **CRAE** | {crae} µm |\n| **CRVE** | {crve} µm |"

    _pendo_track("analysis_pipeline_completed", {
        "crae_value": crae,
        "crve_value": crve,
        "has_vessel_map": skeleton is not None,
        "report_generated_ok": not report.startswith("Ollama Connection Error"),
        "report_length_chars": len(report),
        "crae_within_normal_range": 183 <= crae <= 209,
        "crve_within_normal_range": 205 <= crve <= 235
    })

    return metrics_text, report, skeleton

# --- PENDO INSTALL SCRIPT ---

PENDO_HEAD = """
<script>
(function(apiKey){
    (function(p,e,n,d,o){var v,w,x,y,z;o=p[d]=p[d]||{};o._q=o._q||[];
    v=['initialize','identify','updateOptions','pageLoad','track', 'trackAgent'];for(w=0,x=v.length;w<x;++w)(function(m){
    o[m]=o[m]||function(){o._q[m===v[0]?'unshift':'push']([m].concat([].slice.call(arguments,0)));};})(v[w]);
    y=e.createElement(n);y.async=!0;y.src='https://cdn.pendo.io/agent/static/'+apiKey+'/pendo.js';
    z=e.getElementsByTagName(n)[0];z.parentNode.insertBefore(y,z);})(window,document,'script','pendo');
})('642e678f-029c-471f-ae2d-ca42f76d93fb');

pendo.initialize({
  visitor: {
    id: ''
  }
});
</script>
"""

# --- GRADIO INTERFACE BLOCK ---

with gr.Blocks(head=PENDO_HEAD) as demo:
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
