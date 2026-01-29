import gradio as gr
import torch
import mlflow.pytorch
import numpy as np
from PIL import Image
import json
import requests
from torchvision import transforms
import cv2
from pathlib import Path


# ----------------------------
# CONFIG
# ----------------------------
MODEL_URI = "models:/sugarcane_disease_classifier@production"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

# ----------------------------
# LOAD KNOWLEDGE BASE
# ----------------------------
with open("knowledge_base.json", "r") as f:
    KNOWLEDGE_BASE = json.load(f)

# ----------------------------
# LOAD MODEL FROM MLFLOW
# ----------------------------
print("Loading model from MLflow...")
model = mlflow.pytorch.load_model(MODEL_URI).to(DEVICE)
model.eval()
print("Model loaded.")

# ----------------------------
# IMAGE TRANSFORMS
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# PREDICTION
# ----------------------------
def predict_disease(img: Image.Image):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    return CLASS_NAMES[idx.item()], float(conf.item())

# ----------------------------
# GRAD-CAM (IMPROVED)
# ----------------------------
def generate_gradcam(img, class_idx):
    x = transform(img).unsqueeze(0).to(DEVICE)
    x.requires_grad = True

    features, gradients = [], []

    def forward_hook(_, __, output):
        features.append(output)

    def backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4  # ✅ ResNet18 correct layer
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    out = model(x)
    out[0, class_idx].backward()

    fmap = features[0][0]
    grad = gradients[0].mean(dim=[1, 2])

    cam = torch.zeros(fmap.shape[1:], device=DEVICE)
    for i, w in enumerate(grad):
        cam += w * fmap[i]

    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()

    cam = cv2.GaussianBlur(cam, (21, 21), 0)
    cam = cv2.resize(cam, img.size)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        np.array(img).astype("uint8"),
        0.65,
        heatmap,
        0.35,
        0
    )

    h1.remove()
    h2.remove()
    return overlay


# ----------------------------
# LLM (SAFE)
# ----------------------------
def clean_text(text):
    return (
        text.replace("**", "")
            .replace("*", "")
            .replace("•", "-")
            .strip()
    )

def get_llm_advice(diagnosis):
    """Get advice from LLM based on the diagnosis"""
    try:
        # Prepare the prompt based on whether it's a normal leaf or disease
        if "normal" in diagnosis.lower():
            prompt = """You are a sugarcane cultivation expert. Provide a SINGLE, COMPLETE response for a healthy sugarcane plant. 
            
            RULES:
            - DO NOT ask any questions
            - DO NOT request additional information
            - Provide ALL necessary information in this response
            - Keep it under 250 words
            - Use clear section headers
            - Be specific to sugarcane
            
            FORMAT:
            ✅ **Healthy Sugarcane Plant**
            
            ⚠️ **Watch For**
            - [key warning signs]
            
            Remember: This is a COMPLETE response. Do not ask for more information.
            """
        else:
            prompt = f"""You are an expert in plant pathology. Provide specific care advice for a plant with the following condition: {diagnosis}.
            
            Your response should be factual and practical, including:
            1. A brief description of the condition
            2. Recommended treatment steps
            3. Preventive measures
            4. When to consult a professional
            
            Be concise but thorough in your advice. If you're not certain about the diagnosis, say so."""

        
        # Prepare the request to Ollama
        data = {
            "model": "gemma3",
            "prompt": prompt,
            "stream": False
        }
        
        # Make the request
        response = requests.post("http://localhost:11434/api/generate", json=data)
        response.raise_for_status()
        raw = response.json().get("response", "")
        return clean_text(raw)

    except Exception as e:
        print(f"Error getting advice: {str(e)}")
        return "Error generating advice. Please try again later."


# ----------------------------
# MAIN ANALYSIS
# ----------------------------
def analyze_image(img):
    disease, conf = predict_disease(img)
    idx = CLASS_NAMES.index(disease)

    heatmap = generate_gradcam(img, idx)
    diagnosis_text = f"{disease} (Confidence: {conf*100:.1f}%)"
    advice = get_llm_advice(disease)

    return diagnosis_text, heatmap, advice


def get_disease_info(disease_name):
    """Get information about a disease from the knowledge base"""
    if not disease_name:
        return None
        
    disease_name_lower = disease_name.lower()
    for disease in KNOWLEDGE_BASE['diseases']:
        # Check if the disease name or scientific name contains the search term
        if (disease_name_lower in disease['name'].lower() or 
            disease_name_lower in disease.get('scientific_name', '').lower() or
            any(disease_name_lower in s.lower() for s in disease.get('symptoms', [])) or
            any(disease_name_lower in c.lower() for c in disease.get('causes', []))):
            return disease
    return None

def respond_to_question(question, diagnosis):
    """Answer all questions using KB + LLM only (no hardcoding, no NLP)"""
    if not diagnosis:
        return "Please analyze an image first."

    if not question or len(question.strip()) < 3:
        return "Please ask a clear question."

    try:
        base_diagnosis = diagnosis.split('(')[0].strip()
        disease_info = get_disease_info(base_diagnosis)

        if not disease_info:
            return "No knowledge base information available for this disease."

        context = json.dumps(disease_info, indent=2)

        prompt = f"""
        You are a sugarcane disease expert.

        Use ONLY the information below to answer the question.
        Keep the answer short and direct (max 3 bullet points or 2 sentences).
        Do NOT add new information.
        If the answer is not present, say: "Information not available in knowledge base."

        Disease: {base_diagnosis}
        Knowledge Base:
        {context}

        Question: {question}
        """

        data = {
            "model": "gemma3",
            "prompt": prompt,
            "stream": False,
            "max_tokens": 150
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        return clean_text(raw)

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Error generating response. Please try again."




# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="Crop Health Advisor") as demo:
    gr.Markdown("""
    # 🌱 Crop Health Advisor  
    Upload a **clear, close-up image** of the affected sugarcane leaf or stalk.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Crop Image")
            submit_btn = gr.Button("Analyze")

        with gr.Column():
            output_diagnosis = gr.Textbox(
                label="Diagnosis",
                
            )
            heatmap_output = gr.Image(label="Affected Region (Grad-CAM)")
            output_advice = gr.Textbox(
                label="Care Advice",
                lines=5,
                
            )
        with gr.Accordion("Ask a follow-up question", open=False):
            chat_input = gr.Textbox(
            label="Your question",
            placeholder="Ask about treatment, prevention, severity…"
            )
            chat_output = gr.Textbox(
            label="Response",
            interactive=False
            )
            chat_btn = gr.Button("Ask")


    submit_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[output_diagnosis, heatmap_output, output_advice]
    )
    chat_btn.click(
    fn=respond_to_question,
    inputs=[chat_input, output_diagnosis],
    outputs=chat_output
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
