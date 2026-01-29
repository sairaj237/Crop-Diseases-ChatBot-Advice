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
MODEL_URI = "models:/sugarcane_disease_classifier@staging"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.9

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

    feats, grads = [], []

    def f_hook(_, __, output): feats.append(output)
    def b_hook(_, grad_in, grad_out): grads.append(grad_out[0])

    target_layer = model.blocks[-3]   # 🔑 earlier spatial layer
    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)

    out = model(x)
    out[0, class_idx].backward()

    fmap = feats[0][0]
    grad = grads[0].mean(dim=[1, 2])

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
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3",
                "prompt": f"Treatment and prevention for sugarcane disease: {diagnosis}",
                "stream": False
            }
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        return clean_text(raw)


# ----------------------------
# MAIN ANALYSIS
# ----------------------------
def analyze_image(img):
    disease, conf = predict_disease(img)
    idx = CLASS_NAMES.index(disease)
    heatmap = generate_gradcam(img, idx)

    if conf < CONF_THRESHOLD:
        diagnosis_text = f"Uncertain (Confidence: {conf*100:.1f}%)"
        advice = "⚠️ Model is not confident. Upload a clearer image focusing on the diseased area."
    else:
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
        """Generate a response to the user's question using the knowledge base and LLM"""
        if not diagnosis:
            return "Please analyze an image first so I can provide relevant advice."
        
        try:
            # Get the base diagnosis without the confidence score
            base_diagnosis = diagnosis.split('(')[0].strip()
            
            # If the question is empty or very short, ask for more details
            if not question or len(question.strip()) < 3:
                return "Please ask a specific question about the diagnosis or treatment options."
            
            # Get disease information from knowledge base
            disease_info = get_disease_info(base_diagnosis)
            
            # Common question patterns and their corresponding responses
            question_lower = question.lower()
            
            if disease_info:
                # Handle specific question types using knowledge base
                if any(q in question_lower for q in ['cure', 'treat', 'treatment', 'solution']):
                    if 'treatment' in disease_info:
                        return "\n".join(["✅ Treatment options:"] + [f"• {t}" for t in disease_info['treatment']])
                
                elif any(q in question_lower for q in ['prevent', 'prevention', 'avoid']):
                    if 'prevention' in disease_info:
                        return "\n".join(["🛡️ Prevention measures:"] + [f"• {p}" for p in disease_info['prevention']])
                
                elif any(q in question_lower for q in ['symptom', 'sign', 'look like']):
                    if 'symptoms' in disease_info:
                        return "\n".join(["⚠️ Common symptoms:"] + [f"• {s}" for s in disease_info['symptoms']])
                
                elif any(q in question_lower for q in ['cause', 'reason', 'why']):
                    if 'causes' in disease_info:
                        return "\n".join(["🔍 Possible causes:"] + [f"• {c}" for c in disease_info['causes']])
                
                elif 'curable' in question_lower:
                    status = "Yes" if disease_info.get('is_curable', False) else "No"
                    return f"Curable: {status}. " + ("Early treatment improves success rates." if status == "Yes" else "Focus on prevention and management.")
            
            # If no specific pattern matched or disease not found, use LLM with context
            context = json.dumps(disease_info, indent=2) if disease_info else "No specific information available"
            
            prompt = f"""You are a sugarcane disease expert. Use the following information to answer the question.
            
            Disease: {base_diagnosis}
            Context: {context}
            
            Question: {question}
            
            Guidelines:
            - Be specific and concise (under 150 words)
            - Only use information from the provided context
            - If the question can't be answered from context, say so
            - Do not make up information
            - Format lists with bullet points
            - Do not ask follow-up questions
            """
            
            # Make the API call to Ollama
            data = {
                "model": "gemma3",
                "prompt": prompt,
                "stream": False,
                "max_tokens": 300
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "I couldn't generate a response. Please try again.")
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response. Please try again later."


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
