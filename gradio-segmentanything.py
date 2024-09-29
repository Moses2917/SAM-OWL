import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2

# Load GroundingDINO model using Hugging Face pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# grounding_dino_pipe = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-base", device=device)
OWLv2 = pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16", device=device)
# Load SAM2 model
sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)



def segment_image(image, text_prompt, negative_prompt, mask_all_objects, grounding_threshold):
    # Process image with GroundingDINO
    results = OWLv2(
        image, 
        candidate_labels=[text_prompt],# + ([negative_prompt] if negative_prompt else []),
        threshold=grounding_threshold
    )
    if negative_prompt != []:
        neg_results = OWLv2(
            image,
            candidate_labels= [negative_prompt],
            threshold=grounding_threshold
        )
    if mask_all_objects:
        objects_to_mask = results
    else:
        # Get the object with the highest score
        objects_to_mask = [max(results, key=lambda x: x['score'])]
    
    print(objects_to_mask)
    # Convert image for SAM2
    sam2_image = np.array(image.convert("RGB"))
    color=(255, 0, 0)
    thickness=2
    for im in objects_to_mask:
        start_point = (im['box']['xmin'],im['box']['ymin'])
        end_point = (im['box']['xmax'],im['box']['ymax'])
        sam2_image = cv2.rectangle(sam2_image,start_point,end_point,color, thickness)
    
    sam2_predictor.set_image(sam2_image)
    
    all_masks = []
    
    for obj, neg_oj in zip(objects_to_mask, neg_results):
        box = obj['box']
        if negative_prompt != []:
            neg_box = neg_oj['box']
        # Calculate the center point of the bounding box
        image_width, image_height = image.size
        center_x = (box['xmin'] + box['xmax']) / 2
        center_y = (box['ymin'] + box['ymax']) / 2
        
        new_box = []
        for pt in box:
            new_box.append(box[pt])
        print(new_box)
        box = np.array(new_box)
        if negative_prompt != []:
            neg_center_x = (neg_box['xmin'] + neg_box['xmax']) / 2
            neg_center_y = (neg_box['ymin'] + neg_box['ymax']) / 2
        
        # Generate mask with SAM2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if negative_prompt != []:
                input_point = np.array([[neg_center_x,neg_center_y]])
                input_label = np.array([0])  # 1 is foreground, 0 is background(will be removed)
            else:
                input_point = None
                input_label = None  # 1 is foreground
                
            masks, scores, _ = sam2_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
                box=box
            )

        # Select the mask with the highest score
        best_mask_index = scores.argmax()
        # all_masks.append(masks[best_mask_index])
        all_masks.append(masks)
        
    for mask in all_masks:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # all_masks.append(show_mask(mask))
        
        # all_masks.append()
        

    # # Combine all masks
    # combined_mask = np.logical_or.reduce(all_masks) if all_masks else np.zeros_like(sam2_image[:,:,0], dtype=bool)

    # # Overlay mask on image
    # mask = (combined_mask * 255).astype(np.uint8)
    # masked_image = Image.fromarray(sam2_image).convert("RGBA")
    # mask_image = Image.fromarray(mask).convert("L")
    # masked_image.putalpha(mask_image)
    
    return mask_image
    return masked_image

def segment_image2(image, text_prompt, negative_prompt, mask_all_objects, grounding_threshold):
    results = OWLv2(
        image, 
        candidate_labels=[text_prompt],# + ([negative_prompt] if negative_prompt else []),
        threshold=grounding_threshold
    )
    negative_prompt = [] # just hard coding it off till I get it to work
    neg_boxes = []
    if negative_prompt != []:
        neg_results = OWLv2(
            image,
            candidate_labels= [negative_prompt],
            threshold=grounding_threshold
        )
        # neg_boxes = []
        for neg_result in neg_results:
            # for box_coords in neg_result:
            neg_center_x = (neg_result['box']['xmin'] + neg_result['box']['xmax']) / 2
            neg_center_y = (neg_result['box']['ymin'] + neg_result['box']['ymax']) / 2
            neg_boxes.append([neg_center_x,neg_center_y])
        neg_boxes = np.array(neg_boxes)
        
    if mask_all_objects:
        objects_to_mask = results
    else:
        # Get the object with the highest score
        objects_to_mask = [max(results, key=lambda x: x['score'])]
    
    input_boxes = []
    
    for object in objects_to_mask:
        box = []
        for box_coords in object["box"]:
            box.append(object["box"][box_coords])
        input_boxes.append(box)
    
    input_boxes = np.array(input_boxes)
    
    print(objects_to_mask)
    # Convert image for SAM2
    sam2_image = np.array(image.convert("RGB"))

    
    sam2_predictor.set_image(sam2_image)
    # sam2_predictor.set_image(sam2_image)
    if len(neg_boxes) > 0:
        input_point = neg_boxes
        input_label = np.array([0])
    else:
        input_point = None
        input_label = None
    
    masks, scores, _ = sam2_predictor.predict(
        point_coords=input_point,
        point_labels= input_label,
        box=input_boxes,
        multimask_output=False
    )
    image_width, image_height = image.size
    combined_mask = np.zeros((image_height, image_width, 4), dtype=np.uint8)
    for mask in masks:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        combined_mask = np.maximum(combined_mask, mask_image)
    
    return combined_mask

def detect_obs(image, text_prompt, negative_prompt, grounding_threshold):
    results = OWLv2(
        image, 
        candidate_labels=[text_prompt],# + ([negative_prompt] if negative_prompt else []),
        threshold=grounding_threshold
    )
    sam2_image = np.array(image.convert("RGB"))
    color=(255, 0, 0)
    thickness=2
    for im in results:
        start_point = (im['box']['xmin'],im['box']['ymin'])
        end_point = (im['box']['xmax'],im['box']['ymax'])
        sam2_image = cv2.rectangle(sam2_image,start_point,end_point,color, thickness)
    
    return sam2_image

# Gradio interface
def process_image(input_image, text_prompt, negative_prompt, mask_all_objects, grounding_threshold):
    result = segment_image2(input_image, text_prompt, negative_prompt, mask_all_objects, grounding_threshold)
    detected_objs = detect_obs(input_image, text_prompt,negative_prompt, grounding_threshold)
    return result, detected_objs

iface = gr.Interface(
    # Can add js for a point editor
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Text Prompt"),
        gr.Textbox(label="Negative Prompt (Optional)"),
        gr.Checkbox(label="Mask All Detected Objects"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.12, label="GroundingDINO Threshold"),
    ],
    outputs=[gr.Image(type="pil", label="Segmented Image"), gr.Image(type="numpy", label="Found Objects")],
    title="Image Segmentation with GroundingDINO and SAM2",
    description="Upload an image, provide a text prompt, and optionally a negative prompt to segment the image."
)

iface.launch()